const axios = require("axios");
const LLMProvider = require("./llm.provider");

const AUDIO_SAMPLE_RATE = 16000;
const AUDIO_CHANNELS = 1;
const AUDIO_BITS_PER_SAMPLE = 16;
const LATEST_TURN_SETTLE_MS = 450;
const DEFAULT_VAD_RMS = 650;
const DEFAULT_SILENCE_CHUNKS = 4;
const DEFAULT_MIN_SPEECH_CHUNKS = 2;
const DEFAULT_MAX_SPEECH_CHUNKS = 48;

function safeSend(ws, payload, isBinary = false) {
  if (ws.readyState !== 1) {
    return;
  }

  try {
    ws.send(payload, { binary: isBinary });
  } catch (err) {
    console.error("WebSocket send failed:", err.message);
  }
}

function getEnvUrl(name) {
  return String(process.env[name] || "").trim();
}

function pcmRms(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 2) {
    return 0;
  }

  let sum = 0;
  const sampleCount = Math.floor(buffer.length / 2);

  for (let i = 0; i < sampleCount; i += 1) {
    const sample = buffer.readInt16LE(i * 2);
    sum += sample * sample;
  }

  return Math.sqrt(sum / sampleCount);
}

function buildWavFromPcm(pcmBuffer) {
  const byteRate = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BITS_PER_SAMPLE / 8;
  const blockAlign = AUDIO_CHANNELS * AUDIO_BITS_PER_SAMPLE / 8;
  const header = Buffer.alloc(44);

  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcmBuffer.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(AUDIO_CHANNELS, 22);
  header.writeUInt32LE(AUDIO_SAMPLE_RATE, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(AUDIO_BITS_PER_SAMPLE, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcmBuffer.length, 40);

  return Buffer.concat([header, pcmBuffer]);
}

function stripWavHeaderIfPresent(audioBuffer) {
  if (
    audioBuffer.length > 44 &&
    audioBuffer.toString("ascii", 0, 4) === "RIFF" &&
    audioBuffer.toString("ascii", 8, 12) === "WAVE"
  ) {
    const dataIndex = audioBuffer.indexOf(Buffer.from("data"));
    if (dataIndex >= 0 && dataIndex + 8 < audioBuffer.length) {
      const dataSize = audioBuffer.readUInt32LE(dataIndex + 4);
      const dataStart = dataIndex + 8;
      return audioBuffer.subarray(dataStart, dataStart + dataSize);
    }
  }

  return audioBuffer;
}

function audioBufferFromJson(data) {
  const base64Audio = data?.audio_base64 || data?.audioBase64 || data?.audio || "";
  if (!base64Audio || typeof base64Audio !== "string") {
    return null;
  }

  const cleanBase64 = base64Audio.includes(",")
    ? base64Audio.split(",").pop()
    : base64Audio;
  return Buffer.from(cleanBase64, "base64");
}

async function transcribeLocal(pcmBuffer, signal) {
  const localSttUrl = getEnvUrl("LOCAL_STT_URL");
  if (!localSttUrl) {
    throw new Error("LOCAL_STT_URL is not configured");
  }

  const wavBuffer = buildWavFromPcm(pcmBuffer);
  const res = await axios.post(localSttUrl, wavBuffer, {
    headers: {
      "Content-Type": "audio/wav",
    },
    timeout: Number(process.env.LOCAL_STT_TIMEOUT_MS || 30000),
    signal,
  });

  const text =
    res.data?.text ||
    res.data?.transcript ||
    res.data?.result ||
    "";

  return String(text).trim();
}

async function synthesizeLocal(text, signal) {
  const localTtsUrl = getEnvUrl("LOCAL_TTS_URL");
  if (!localTtsUrl) {
    throw new Error("LOCAL_TTS_URL is not configured");
  }

  const res = await axios.post(
    localTtsUrl,
    {
      text,
      sample_rate: AUDIO_SAMPLE_RATE,
      format: "pcm_s16le",
    },
    {
      responseType: "arraybuffer",
      timeout: Number(process.env.LOCAL_TTS_TIMEOUT_MS || 30000),
      signal,
      validateStatus: (status) => status >= 200 && status < 300,
    }
  );

  const contentType = String(res.headers?.["content-type"] || "");
  let audioBuffer;

  if (contentType.includes("application/json")) {
    const jsonText = Buffer.from(res.data).toString("utf8");
    audioBuffer = audioBufferFromJson(JSON.parse(jsonText));
  } else {
    audioBuffer = Buffer.from(res.data);
  }

  if (!audioBuffer || audioBuffer.length === 0) {
    throw new Error("Local TTS returned no audio");
  }

  return stripWavHeaderIfPresent(audioBuffer);
}

module.exports = function handleVoiceSession(ws) {
  const llm = new LLMProvider();
  const sessionUserId = `voice_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const history = [];

  const vadRms = Number(process.env.LOCAL_STS_VAD_RMS || DEFAULT_VAD_RMS);
  const silenceChunksNeeded = Number(process.env.LOCAL_STS_SILENCE_CHUNKS || DEFAULT_SILENCE_CHUNKS);
  const minSpeechChunks = Number(process.env.LOCAL_STS_MIN_SPEECH_CHUNKS || DEFAULT_MIN_SPEECH_CHUNKS);
  const maxSpeechChunks = Number(process.env.LOCAL_STS_MAX_SPEECH_CHUNKS || DEFAULT_MAX_SPEECH_CHUNKS);

  let currentTurn = null;
  let queuedTurn = null;
  let nextTurnId = 0;
  let isClosed = false;
  let queueTimer = null;
  let speechChunks = [];
  let silenceChunks = 0;
  let isCapturingSpeech = false;

  function sendEvent(type, extra = {}) {
    safeSend(ws, JSON.stringify({ type, ...extra }));
  }

  function sendStatus(state, extra = {}) {
    sendEvent("status", { state, ...extra });
  }

  function clearQueueTimer() {
    if (queueTimer) {
      clearTimeout(queueTimer);
      queueTimer = null;
    }
  }

  function resetSpeechBuffer() {
    speechChunks = [];
    silenceChunks = 0;
    isCapturingSpeech = false;
  }

  function scheduleQueueProcessing(delayMs = LATEST_TURN_SETTLE_MS) {
    clearQueueTimer();
    queueTimer = setTimeout(() => {
      queueTimer = null;
      queueMicrotask(processQueue);
    }, delayMs);
  }

  async function interruptCurrentTurn(reason = "interrupt") {
    if (currentTurn?.abortController) {
      currentTurn.abortController.abort();
    }

    if (currentTurn?.historyInjected) {
      const last = history[history.length - 1];
      if (last?.role === "user" && last.content === currentTurn.text) {
        history.pop();
      }
      currentTurn.historyInjected = false;
    }

    if (currentTurn?.stage === "speaking") {
      sendStatus("interrupted", { reason, turnId: currentTurn.id });
    }

    if (currentTurn) {
      currentTurn.cancelled = true;
    }
  }

  async function processQueue() {
    if (currentTurn || !queuedTurn || isClosed) {
      return;
    }

    currentTurn = queuedTurn;
    queuedTurn = null;
    const turn = currentTurn;

    sendStatus("thinking", { turnId: turn.id, text: turn.text });

    const abortController = new AbortController();
    turn.abortController = abortController;
    turn.stage = "thinking";
    turn.historyInjected = true;
    history.push({ role: "user", content: turn.text });

    let replyText = "";

    try {
      replyText = await llm.generateReply(turn.text, history, sessionUserId, {
        signal: abortController.signal,
      });
    } catch (err) {
      if (err?.code !== "ERR_CANCELED") {
        console.error("LLM error:", err.message);
        sendEvent("bot_response", {
          text: "Sorry, I ran into a problem thinking about that.",
          turnId: turn.id,
        });
      }
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    if (turn.cancelled || currentTurn?.id !== turn.id || isClosed) {
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    const trimmedReply = String(replyText || "").trim();
    if (!trimmedReply) {
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    history.push({ role: "assistant", content: trimmedReply });
    sendEvent("bot_response", { text: trimmedReply, turnId: turn.id });

    turn.stage = "speaking";
    sendStatus("speaking", { turnId: turn.id, text: trimmedReply });

    try {
      const audioBuffer = await synthesizeLocal(trimmedReply, abortController.signal);
      if (!turn.cancelled && currentTurn?.id === turn.id && !isClosed) {
        safeSend(ws, audioBuffer, true);
      }
    } catch (err) {
      if (!turn.cancelled) {
        console.error("Local TTS failed:", err.message);
        sendStatus("tts_error", {
          detail: "Local TTS failed. Text response was still delivered.",
          turnId: turn.id,
        });
      }
    }

    if (!turn.cancelled && !isClosed) {
      sendStatus("listening", { turnId: turn.id });
    }

    currentTurn = null;
    scheduleQueueProcessing(0);
  }

  function enqueueTurn(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed || isClosed) {
      return;
    }

    const turn = {
      id: ++nextTurnId,
      text: trimmed,
      cancelled: false,
      stage: "queued",
      abortController: null,
      historyInjected: false,
    };

    sendEvent("final", { text: trimmed, turnId: turn.id });

    if (currentTurn) {
      queuedTurn = turn;
      void interruptCurrentTurn("user-barge-in");
      scheduleQueueProcessing();
      return;
    }

    queuedTurn = turn;
    sendStatus("listening", { reason: "queued-latest", turnId: turn.id });
    scheduleQueueProcessing();
  }

  async function finalizeSpeechBuffer() {
    if (speechChunks.length < minSpeechChunks) {
      resetSpeechBuffer();
      return;
    }

    const pcmBuffer = Buffer.concat(speechChunks);
    resetSpeechBuffer();
    sendStatus("transcribing");

    try {
      const transcript = await transcribeLocal(pcmBuffer);
      if (!transcript || isClosed) {
        sendStatus("listening");
        return;
      }

      sendEvent("partial", {
        text: transcript,
        turnId: currentTurn?.id || queuedTurn?.id || nextTurnId + 1,
      });
      enqueueTurn(transcript);
    } catch (err) {
      console.error("Local STT failed:", err.message);
      sendStatus("error", {
        detail: "Local STT failed. Check LOCAL_STT_URL and your local speech server.",
      });
    }
  }

  function handleAudioChunk(data) {
    if (currentTurn || queuedTurn || isClosed) {
      return;
    }

    const chunk = Buffer.from(data);
    const rms = pcmRms(chunk);
    const hasSpeech = rms >= vadRms;

    if (hasSpeech) {
      isCapturingSpeech = true;
      silenceChunks = 0;
      speechChunks.push(chunk);
      sendStatus("listening", { level: Math.round(rms) });

      if (speechChunks.length >= maxSpeechChunks) {
        void finalizeSpeechBuffer();
      }
      return;
    }

    if (!isCapturingSpeech) {
      return;
    }

    speechChunks.push(chunk);
    silenceChunks += 1;

    if (silenceChunks >= silenceChunksNeeded) {
      void finalizeSpeechBuffer();
    }
  }

  sendStatus("listening");

  ws.on("message", async (data, isBinary) => {
    if (!isBinary) {
      try {
        const raw = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
        const msg = JSON.parse(raw);

        if (msg.type === "interrupt") {
          resetSpeechBuffer();
          await interruptCurrentTurn("frontend-interrupt");
          sendStatus("listening", { reason: "interrupt" });
        }
      } catch (err) {
        console.error("Invalid control message:", err.message);
      }
      return;
    }

    handleAudioChunk(data);
  });

  ws.on("close", async () => {
    isClosed = true;
    clearQueueTimer();
    queuedTurn = null;
    resetSpeechBuffer();
    await interruptCurrentTurn("socket-close");
  });
};
