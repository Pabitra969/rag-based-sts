import os
import io
import subprocess
import tempfile
import wave
import audioop
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import onnxruntime
    from piper.config import PiperConfig, SynthesisConfig
    from piper.voice import PiperVoice
except Exception:
    onnxruntime = None
    PiperConfig = None
    SynthesisConfig = None
    PiperVoice = None


APP_HOST = os.environ.get("LOCAL_SPEECH_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("LOCAL_SPEECH_PORT", "7001"))
SAMPLE_RATE = 16000
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "faster-whisper-small.en"
DEFAULT_PIPER_DIR = Path(__file__).resolve().parent / "models" / "piper"
DEFAULT_PIPER_COREML_CACHE_DIR = DEFAULT_PIPER_DIR / ".coreml-cache"

app = FastAPI(title="Local Offline Speech Server")
_whisper_model = None
_piper_voice = None


class SpeakRequest(BaseModel):
    text: str
    sample_rate: int = SAMPLE_RATE
    format: str = "pcm_s16le"


def get_whisper_model():
    global _whisper_model

    if _whisper_model is not None:
      return _whisper_model

    if WhisperModel is None:
        return None

    model_path = os.environ.get("WHISPER_MODEL_PATH", "").strip()
    if not model_path and DEFAULT_MODEL_PATH.exists():
        model_path = str(DEFAULT_MODEL_PATH)

    if not model_path:
        return None

    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    _whisper_model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )
    return _whisper_model


def command_exists(command):
    return subprocess.run(
        ["which", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def resolve_piper_binary():
    configured = os.environ.get("PIPER_BIN", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return str(candidate)
        return configured

    return "piper" if command_exists("piper") else ""


def resolve_piper_model_path():
    configured = os.environ.get("PIPER_MODEL_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate
        return None

    if DEFAULT_PIPER_DIR.exists():
        matches = sorted(DEFAULT_PIPER_DIR.glob("*.onnx"))
        if matches:
            return matches[0]

    return None


def resolve_piper_config_path(model_path):
    configured = os.environ.get("PIPER_CONFIG_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return candidate
        return None

    if not model_path:
        return None

    sibling = Path(f"{model_path}.json")
    if sibling.exists():
        return sibling

    alt = model_path.with_suffix(model_path.suffix + ".json")
    if alt.exists():
        return alt

    return None


def piper_is_available():
    return bool(PiperVoice is not None and onnxruntime is not None and resolve_piper_model_path())


def resolve_piper_providers():
    if onnxruntime is None:
        return ["CPUExecutionProvider"]

    available = set(onnxruntime.get_available_providers())
    acceleration = os.environ.get("PIPER_ACCELERATION", "cpu").strip().lower()
    cache_dir = Path(
        os.environ.get("PIPER_COREML_CACHE_DIR", str(DEFAULT_PIPER_COREML_CACHE_DIR))
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if acceleration in {"auto", "coreml", "gpu"} and "CoreMLExecutionProvider" in available:
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "0",
                    "ModelCacheDirectory": str(cache_dir),
                },
            ),
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


def get_piper_voice():
    global _piper_voice

    if _piper_voice is not None:
        return _piper_voice

    if not piper_is_available():
        return None

    model_path = resolve_piper_model_path()
    config_path = resolve_piper_config_path(model_path)
    if not model_path or not config_path or not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    providers = resolve_piper_providers()
    session = None
    last_error = None

    for provider_set in (providers, ["CPUExecutionProvider"]):
        try:
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=provider_set,
            )
            break
        except Exception as err:
            last_error = err
            session = None

    if session is None:
        raise RuntimeError(f"Failed to initialize Piper session: {last_error}")

    _piper_voice = PiperVoice(
        session=session,
        config=PiperConfig.from_dict(config_dict),
        download_dir=model_path.parent,
    )
    return _piper_voice


def get_piper_synthesis_config():
    if SynthesisConfig is None:
        return None

    speaker = os.environ.get("PIPER_SPEAKER", "").strip()
    speaker_id = int(speaker) if speaker.isdigit() else None

    return SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=float(os.environ.get("PIPER_LENGTH_SCALE", "1.0")),
        noise_scale=float(os.environ.get("PIPER_NOISE_SCALE", "0.667")),
        noise_w_scale=float(os.environ.get("PIPER_NOISE_W", "0.8")),
        volume=float(os.environ.get("PIPER_VOLUME", "1.0")),
    )


def synthesize_with_piper(text, sample_rate):
    voice = get_piper_voice()
    if voice is None:
        raise RuntimeError("Piper is not available")

    syn_config = get_piper_synthesis_config()
    pcm_chunks = []
    source_rate = None

    for audio_chunk in voice.synthesize(text, syn_config=syn_config):
        source_rate = source_rate or audio_chunk.sample_rate
        pcm_chunks.append(audio_chunk.audio_int16_bytes)

    pcm_bytes = b"".join(pcm_chunks)
    if not pcm_bytes or not source_rate:
        raise RuntimeError("Piper produced no audio")

    target_rate = int(sample_rate or SAMPLE_RATE)
    if target_rate != source_rate:
        pcm_bytes, _state = audioop.ratecv(
            pcm_bytes,
            2,
            1,
            source_rate,
            target_rate,
            None,
        )
        source_rate = target_rate

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(source_rate)
        wav_file.writeframes(pcm_bytes)

    return wav_buffer.getvalue()


def get_tts_backend():
    if piper_is_available():
        return "piper"
    if command_exists("say"):
        return "say"
    return None


@app.get("/health")
def health():
    stt_available = get_whisper_model() is not None
    tts_backend = get_tts_backend()
    tts_available = tts_backend is not None
    status_code = 200 if stt_available and tts_available else 503

    return Response(
        content=(
            "{"
            f"\"service\":true,"
            f"\"stt_available\":{str(stt_available).lower()},"
            f"\"tts_available\":{str(tts_available).lower()},"
            f"\"tts_backend\":\"{tts_backend or ''}\""
            "}"
        ),
        media_type="application/json",
        status_code=status_code,
    )


@app.post("/transcribe")
async def transcribe(request: Request):
    model = get_whisper_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Offline STT is not configured. Install faster-whisper and set "
                "WHISPER_MODEL_PATH to a local model directory."
            ),
        )

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio body is required")

    content_type = str(request.headers.get("content-type", "")).lower()
    if "webm" in content_type:
        suffix = ".webm"
    elif "mpeg" in content_type or "mp3" in content_type:
        suffix = ".mp3"
    elif "wav" in content_type:
        suffix = ".wav"
    else:
        suffix = ".wav"

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / f"speech{suffix}"
        audio_path.write_bytes(audio)
        segments, _info = model.transcribe(
            str(audio_path),
            language="en",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 120,
            },
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()

    return {"text": text}


@app.post("/speak")
async def speak(payload: SpeakRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    tts_backend = get_tts_backend()
    if tts_backend is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No local TTS backend is available. Configure Piper with "
                "PIPER_BIN and PIPER_MODEL_PATH, or ensure macOS say is available."
            ),
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "speech.wav"
        if tts_backend == "piper":
            return Response(
                synthesize_with_piper(text, payload.sample_rate or SAMPLE_RATE),
                media_type="audio/wav",
            )
        else:
            say_voice = os.environ.get("LOCAL_TTS_SAY_VOICE", "").strip()
            say_rate = os.environ.get("LOCAL_TTS_SAY_RATE", "").strip()
            say_cmd = ["say"]
            if say_voice:
                say_cmd.extend(["-v", say_voice])
            if say_rate:
                say_cmd.extend(["-r", say_rate])
            say_cmd.extend(
                [
                    "-o",
                    str(wav_path),
                    "--file-format=WAVE",
                    f"--data-format=LEI16@{payload.sample_rate or SAMPLE_RATE}",
                    text,
                ]
            )
            subprocess.run(say_cmd, check=True)

        return Response(wav_path.read_bytes(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
