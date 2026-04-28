const axios = require("axios");
const LLMProvider = require("./llm.provider");

const AUDIO_SAMPLE_RATE = 16000;
const AUDIO_CHANNELS = 1;
const AUDIO_BITS_PER_SAMPLE = 16;
const LATEST_TURN_SETTLE_MS = 450;
const DEFAULT_VAD_RMS = 350;
const DEFAULT_VAD_NOISE_MULTIPLIER = 2.45;
const DEFAULT_SILENCE_CHUNKS = 2;
const DEFAULT_MIN_SPEECH_CHUNKS = 1;
const DEFAULT_MAX_SPEECH_CHUNKS = 48;
const DEFAULT_PREROLL_CHUNKS = 3;
const DEFAULT_TRIGGER_SPEECH_CHUNKS = 2;
const DEFAULT_MIN_SPEECH_RATIO = 0.38;
const DEFAULT_MIN_UTTERANCE_MS = 320;
const DEFAULT_MIN_ZCR = 0.015;
const DEFAULT_MAX_ZCR = 0.24;
const DEFAULT_NEAR_FIELD_RMS_MULTIPLIER = 1.15;
const DEFAULT_NEAR_FIELD_PEAK_MULTIPLIER = 1.7;
const DEFAULT_KEYBOARD_MAX_ZCR = 0.19;
const DEFAULT_KEYBOARD_PEAK_RATIO = 5.2;
const TTS_CHUNK_MAX_CHARS = Number(process.env.LOCAL_STS_TTS_CHUNK_MAX_CHARS || 90);
const TTS_CHUNK_MIN_SENTENCES = Number(process.env.LOCAL_STS_TTS_CHUNK_MIN_SENTENCES || 1);
const TTS_EARLY_START_CHARS = Number(process.env.LOCAL_STS_TTS_EARLY_START_CHARS || 42);

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

function pcmStats(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 4) {
    return { rms: 0, peak: 0, zcr: 0 };
  }

  let sum = 0;
  let peak = 0;
  let zeroCrossings = 0;
  const sampleCount = Math.floor(buffer.length / 2);
  let prev = buffer.readInt16LE(0);

  for (let i = 0; i < sampleCount; i += 1) {
    const sample = buffer.readInt16LE(i * 2);
    const absSample = Math.abs(sample);
    sum += sample * sample;
    peak = Math.max(peak, absSample);

    if ((sample >= 0 && prev < 0) || (sample < 0 && prev >= 0)) {
      zeroCrossings += 1;
    }
    prev = sample;
  }

  return {
    rms: Math.sqrt(sum / sampleCount),
    peak,
    zcr: zeroCrossings / sampleCount,
  };
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

function splitReadySegments(buffer, isFinal = false) {
  const parts = [];
  let remaining = buffer;
  const sentenceRegex = /(.+?[.!?,;:])(\s+|$)/;

  while (true) {
    const match = remaining.match(sentenceRegex);
    if (!match) {
      break;
    }

    parts.push(match[1].trim());
    remaining = remaining.slice(match.index + match[0].length);
  }

  if (isFinal && remaining.trim()) {
    parts.push(remaining.trim());
    remaining = "";
  }

  return { parts, remaining };
}

function chunkForSpeech(text, isFinal = false) {
  const cleaned = String(text || "").trim();
  if (!cleaned) {
    return { parts: [], remaining: "" };
  }

  const { parts, remaining } = splitReadySegments(cleaned, isFinal);
  const output = [];
  let buffer = "";

  for (const part of parts) {
    const candidate = buffer ? `${buffer} ${part}` : part;
    if (candidate.length > TTS_CHUNK_MAX_CHARS && buffer) {
      output.push(buffer);
      buffer = part;
      continue;
    }

    buffer = candidate;
    const sentenceCount = (buffer.match(/[.!?,;:](\s|$)/g) || []).length;
    if (sentenceCount >= TTS_CHUNK_MIN_SENTENCES || buffer.length >= TTS_CHUNK_MAX_CHARS) {
      output.push(buffer);
      buffer = "";
    }
  }

  if (isFinal) {
    const finalText = [buffer, remaining].filter(Boolean).join(" ").trim();
    if (finalText) {
      output.push(finalText);
    }
    return { parts: output, remaining: "" };
  }

  if (!output.length && cleaned.length >= TTS_EARLY_START_CHARS) {
    const cutIndex = cleaned.lastIndexOf(" ", TTS_EARLY_START_CHARS);
    if (cutIndex > 18) {
      return {
        parts: [cleaned.slice(0, cutIndex).trim()],
        remaining: cleaned.slice(cutIndex + 1).trim(),
      };
    }
  }

  return {
    parts: output,
    remaining: [buffer, remaining].filter(Boolean).join(" ").trim(),
  };
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
  let ttsMode = "server";
  let sttMode = "server";

  const vadRms = Number(process.env.LOCAL_STS_VAD_RMS || DEFAULT_VAD_RMS);
  const vadNoiseMultiplier = Number(
    process.env.LOCAL_STS_VAD_NOISE_MULTIPLIER || DEFAULT_VAD_NOISE_MULTIPLIER
  );
  const silenceChunksNeeded = Number(process.env.LOCAL_STS_SILENCE_CHUNKS || DEFAULT_SILENCE_CHUNKS);
  const minSpeechChunks = Number(process.env.LOCAL_STS_MIN_SPEECH_CHUNKS || DEFAULT_MIN_SPEECH_CHUNKS);
  const maxSpeechChunks = Number(process.env.LOCAL_STS_MAX_SPEECH_CHUNKS || DEFAULT_MAX_SPEECH_CHUNKS);
  const prerollChunksLimit = Number(process.env.LOCAL_STS_PREROLL_CHUNKS || DEFAULT_PREROLL_CHUNKS);
  const triggerSpeechChunks = Number(
    process.env.LOCAL_STS_TRIGGER_SPEECH_CHUNKS || DEFAULT_TRIGGER_SPEECH_CHUNKS
  );
  const minSpeechRatio = Number(process.env.LOCAL_STS_MIN_SPEECH_RATIO || DEFAULT_MIN_SPEECH_RATIO);
  const minUtteranceMs = Number(process.env.LOCAL_STS_MIN_UTTERANCE_MS || DEFAULT_MIN_UTTERANCE_MS);
  const minSpeechZcr = Number(process.env.LOCAL_STS_MIN_SPEECH_ZCR || DEFAULT_MIN_ZCR);
  const maxSpeechZcr = Number(process.env.LOCAL_STS_MAX_SPEECH_ZCR || DEFAULT_MAX_ZCR);
  const nearFieldRmsMultiplier = Number(
    process.env.LOCAL_STS_NEAR_FIELD_RMS_MULTIPLIER || DEFAULT_NEAR_FIELD_RMS_MULTIPLIER
  );
  const nearFieldPeakMultiplier = Number(
    process.env.LOCAL_STS_NEAR_FIELD_PEAK_MULTIPLIER || DEFAULT_NEAR_FIELD_PEAK_MULTIPLIER
  );
  const keyboardMaxZcr = Number(process.env.LOCAL_STS_KEYBOARD_MAX_ZCR || DEFAULT_KEYBOARD_MAX_ZCR);
  const keyboardPeakRatio = Number(
    process.env.LOCAL_STS_KEYBOARD_PEAK_RATIO || DEFAULT_KEYBOARD_PEAK_RATIO
  );

  let currentTurn = null;
  let queuedTurn = null;
  let nextTurnId = 0;
  let isClosed = false;
  let queueTimer = null;
  let speechChunks = [];
  let prerollChunks = [];
  let silenceChunks = 0;
  let isCapturingSpeech = false;
  let speechStreak = 0;
  let noiseFloorRms = Math.max(80, Math.round(vadRms * 0.45));
  let captureStats = null;

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
    prerollChunks = [];
    silenceChunks = 0;
    isCapturingSpeech = false;
    speechStreak = 0;
    captureStats = null;
  }

  function rememberPreroll(chunk) {
    prerollChunks.push(Buffer.from(chunk));
    if (prerollChunks.length > prerollChunksLimit) {
      prerollChunks.shift();
    }
  }

  function updateNoiseFloor(rms) {
    if (!Number.isFinite(rms) || rms <= 0) {
      return;
    }

    if (rms < noiseFloorRms * 1.5) {
      noiseFloorRms = Math.max(35, Math.round((noiseFloorRms * 0.88) + (rms * 0.12)));
    }
  }

  function isSpeechLike(stats, dynamicThreshold) {
    const hasEnergy = stats.rms >= dynamicThreshold;
    const plausibleVoiceBand = stats.zcr >= minSpeechZcr && stats.zcr <= maxSpeechZcr;
    const hasPeak = stats.peak >= Math.max(dynamicThreshold * 1.55, vadRms * 1.15);
    const peakRatio = stats.rms > 0 ? stats.peak / stats.rms : 0;
    const keyboardLike = peakRatio >= keyboardPeakRatio || stats.zcr > keyboardMaxZcr;
    const nearFieldLike =
      stats.rms >= noiseFloorRms * nearFieldRmsMultiplier &&
      stats.peak >= dynamicThreshold * nearFieldPeakMultiplier;
    return hasEnergy && plausibleVoiceBand && hasPeak && !keyboardLike && nearFieldLike;
  }

  function startSpeechCapture(initialStats) {
    isCapturingSpeech = true;
    silenceChunks = 0;
    speechChunks = prerollChunks.slice();
    prerollChunks = [];
    captureStats = {
      totalChunks: Math.max(speechChunks.length, 1),
      speechLikeChunks: speechStreak,
      peakRms: initialStats.rms,
      latestThreshold: Math.max(vadRms, noiseFloorRms * vadNoiseMultiplier),
    };
  }

  function recordCaptureChunk(isSpeech, stats) {
    if (!captureStats) {
      captureStats = {
        totalChunks: 0,
        speechLikeChunks: 0,
        peakRms: 0,
        latestThreshold: Math.max(vadRms, noiseFloorRms * vadNoiseMultiplier),
      };
    }

    captureStats.totalChunks += 1;
    captureStats.peakRms = Math.max(captureStats.peakRms, stats.rms);
    captureStats.latestThreshold = Math.max(vadRms, noiseFloorRms * vadNoiseMultiplier);
    if (isSpeech) {
      captureStats.speechLikeChunks += 1;
    }
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
    sendEvent("turn_started", { turnId: turn.id, text: turn.text });

    sendStatus("thinking", { turnId: turn.id, text: turn.text });

    const abortController = new AbortController();
    turn.abortController = abortController;
    turn.stage = "thinking";
    turn.historyInjected = true;
    history.push({ role: "user", content: turn.text });

    let replyText = "";
    let startedSpeaking = false;
    let speechBuffer = "";
    let hasSpokenChunks = false;
    let hasQueuedSpeechParts = false;
    let speechWorkerFailed = null;
    let speechWorkerWake = null;
    let speechQueueClosed = false;
    const speechQueue = [];

    function wakeSpeechWorker() {
      if (speechWorkerWake) {
        speechWorkerWake();
        speechWorkerWake = null;
      }
    }

    function pushSpeechPart(part) {
      const trimmed = String(part || "").trim();
      if (!trimmed || turn.cancelled || currentTurn?.id !== turn.id || isClosed) {
        return;
      }

      hasQueuedSpeechParts = true;
      speechQueue.push(trimmed);
      wakeSpeechWorker();
    }

    function closeSpeechQueue() {
      speechQueueClosed = true;
      wakeSpeechWorker();
    }

    async function runSpeechWorker() {
      while (!isClosed) {
        if (turn.cancelled || currentTurn?.id !== turn.id) {
          return;
        }

        if (!speechQueue.length) {
          if (speechQueueClosed) {
            return;
          }
          await new Promise((resolve) => {
            speechWorkerWake = resolve;
          });
          continue;
        }

        const part = speechQueue.shift();
        if (!part) {
          continue;
        }

        if (!startedSpeaking) {
          turn.stage = "speaking";
          sendStatus("speaking", { turnId: turn.id, text: part });
          startedSpeaking = true;
        }

        if (ttsMode === "browser") {
          sendEvent("bot_tts", { text: part, turnId: turn.id });
          hasSpokenChunks = true;
          continue;
        }

        try {
          const audioBuffer = await synthesizeLocal(part, abortController.signal);
          if (!turn.cancelled && currentTurn?.id === turn.id && !isClosed) {
            safeSend(ws, audioBuffer, true);
            hasSpokenChunks = true;
          }
        } catch (err) {
          speechWorkerFailed = err;
          if (!turn.cancelled) {
            abortController.abort();
          }
          return;
        }
      }
    }

    const speechWorkerPromise = runSpeechWorker();

    try {
      for await (const chunk of llm.streamReply(turn.text, history, sessionUserId, {
        signal: abortController.signal,
      })) {
        if (turn.cancelled || currentTurn?.id !== turn.id || isClosed) {
          currentTurn = null;
          scheduleQueueProcessing(0);
          return;
        }

        replyText += chunk;
        sendEvent("bot_partial", { text: replyText, turnId: turn.id });

        speechBuffer += chunk;
        const bufferedChunks = chunkForSpeech(speechBuffer, false);
        speechBuffer = bufferedChunks.remaining;

        for (const part of bufferedChunks.parts) {
          pushSpeechPart(part);
        }

        if (speechWorkerFailed) {
          throw speechWorkerFailed;
        }
      }
    } catch (err) {
      closeSpeechQueue();
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

    try {
      const finalChunks = chunkForSpeech(speechBuffer, true).parts;
      const speechParts = finalChunks.length
        ? finalChunks
        : ((hasSpokenChunks || hasQueuedSpeechParts) ? [] : [trimmedReply]);

      for (const part of speechParts) {
        pushSpeechPart(part);
      }
      closeSpeechQueue();
      await speechWorkerPromise;
      if (speechWorkerFailed) {
        throw speechWorkerFailed;
      }
    } catch (err) {
      closeSpeechQueue();
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

  function enqueueTurn(text, options = {}) {
    const { emitFinal = true } = options;
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

    if (emitFinal) {
      sendEvent("final", { text: trimmed, turnId: turn.id });
    }

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
    const totalChunks = captureStats?.totalChunks || speechChunks.length;
    const speechLikeChunks = captureStats?.speechLikeChunks || 0;
    const speechRatio = totalChunks > 0 ? speechLikeChunks / totalChunks : 0;
    const utteranceDurationMs = Math.round((pcmBuffer.length / 2 / AUDIO_SAMPLE_RATE) * 1000);
    const utterancePeak = captureStats?.peakRms || pcmRms(pcmBuffer);
    const activeThreshold = captureStats?.latestThreshold || Math.max(vadRms, noiseFloorRms * vadNoiseMultiplier);
    resetSpeechBuffer();

    if (
      utteranceDurationMs < minUtteranceMs ||
      speechLikeChunks < triggerSpeechChunks ||
      speechRatio < minSpeechRatio ||
      utterancePeak < Math.max(activeThreshold * 1.02, vadRms * 1.08)
    ) {
      sendStatus("listening", { reason: "filtered-noise" });
      return;
    }

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
    const stats = pcmStats(chunk);
    const dynamicThreshold = Math.max(vadRms, noiseFloorRms * vadNoiseMultiplier);
    const relaxedThreshold = Math.max(vadRms * 0.8, noiseFloorRms * Math.max(1.85, vadNoiseMultiplier - 0.3));
    const hasSpeech = isSpeechLike(stats, dynamicThreshold);
    const keepAliveSpeech = isSpeechLike(stats, relaxedThreshold);

    if (hasSpeech) {
      rememberPreroll(chunk);
      speechStreak += 1;

      if (!isCapturingSpeech) {
        if (speechStreak < triggerSpeechChunks) {
          return;
        }
        startSpeechCapture(stats);
      } else {
        speechChunks.push(chunk);
        recordCaptureChunk(true, stats);
      }

      silenceChunks = 0;
      sendStatus("listening", { level: Math.round(stats.rms) });

      if (speechChunks.length >= maxSpeechChunks) {
        void finalizeSpeechBuffer();
      }
      return;
    }

    speechStreak = 0;

    if (!isCapturingSpeech) {
      updateNoiseFloor(stats.rms);
      rememberPreroll(chunk);
      return;
    }

    speechChunks.push(chunk);
    silenceChunks += 1;
    recordCaptureChunk(keepAliveSpeech, stats);

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
        } else if (msg.type === "session_config") {
          ttsMode = msg.ttsMode === "browser" ? "browser" : "server";
          sttMode = msg.sttMode === "browser" ? "browser" : "server";
        } else if (msg.type === "voice_text_final") {
          resetSpeechBuffer();
          enqueueTurn(msg.text, { emitFinal: false });
        }
      } catch (err) {
        console.error("Invalid control message:", err.message);
      }
      return;
    }

    if (sttMode === "browser") {
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
