import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


APP_HOST = os.environ.get("LOCAL_SPEECH_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("LOCAL_SPEECH_PORT", "7001"))
SAMPLE_RATE = 16000
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "faster-whisper-small.en"

app = FastAPI(title="Local Offline Speech Server")
_whisper_model = None


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


@app.get("/health")
def health():
    stt_available = get_whisper_model() is not None
    tts_available = command_exists("say") and command_exists("afconvert")
    status_code = 200 if stt_available and tts_available else 503

    return Response(
        content=(
            "{"
            f"\"service\":true,"
            f"\"stt_available\":{str(stt_available).lower()},"
            f"\"tts_available\":{str(tts_available).lower()}"
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

    if not command_exists("say") or not command_exists("afconvert"):
        raise HTTPException(
            status_code=503,
            detail="macOS say/afconvert are required for local TTS fallback.",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        aiff_path = Path(tmp_dir) / "speech.aiff"
        wav_path = Path(tmp_dir) / "speech.wav"

        subprocess.run(["say", "-o", str(aiff_path), text], check=True)
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{payload.sample_rate or SAMPLE_RATE}",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )

        return Response(wav_path.read_bytes(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
