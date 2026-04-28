const VAD_THRESHOLD = 0.0035;
const AUDIO_SAMPLE_RATE = 16000;
const DEFAULT_WS = "ws://localhost:5005/ws/voice";
const WEBSOCKET_URL = window.VOICE_WS_URL || DEFAULT_WS;
const STT_MODE = window.VOICE_STT_MODE || (
  (window.SpeechRecognition || window.webkitSpeechRecognition) ? "browser" : "server"
);
const TTS_MODE = window.VOICE_TTS_MODE || ("speechSynthesis" in window ? "browser" : "server");
const BROWSER_STT_COMMIT_DEBOUNCE_MS = 180;
const BROWSER_STT_IDLE_COMMIT_MS = 700;
const BROWSER_STT_RESUME_AFTER_TTS_MS = 450;
const BROWSER_STT_WATCHDOG_MS = 2200;
const VAD_NOISE_MULTIPLIER = 2.35;
const VAD_TRIGGER_FRAMES = 2;
const VAD_HANGOVER_FRAMES = 2;
const VAD_MAX_PREROLL_FRAMES = 3;
const VAD_MIN_ZCR = 0.015;
const VAD_MAX_ZCR = 0.24;
const NEAR_FIELD_PEAK_MULTIPLIER = 1.75;
const NEAR_FIELD_RMS_MULTIPLIER = 1.2;
const KEYBOARD_MAX_ZCR = 0.19;
const KEYBOARD_PEAK_RATIO = 5.5;

let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let voiceSocket;
let isVoiceModeActive = false;
let currentAudioSource = null;
let playbackStateTimer = null;
let currentVoiceState = "idle";
let audioQueue = [];
let isPlayingQueue = false;
let browserTtsQueue = [];
let browserTtsActive = false;
let browserTtsCurrent = null;
let pendingListeningPayload = null;
let browserRecognition = null;
let browserRecognitionRunning = false;
let browserRecognitionShouldRun = false;
let browserRecognitionRestartTimer = null;
let browserSttCommitTimer = null;
let browserSttIdleTimer = null;
let browserSttResumeTimer = null;
let browserSttWatchdogTimer = null;
let browserFinalTranscript = "";
let browserLatestInterim = "";
let activeAssistantTurnId = null;
let activeSttMode = "server";
let activeTtsMode = "server";
let adaptiveNoiseFloor = VAD_THRESHOLD * 0.55;
let speechFrameStreak = 0;
let silenceFrameStreak = 0;
let isDetectorOpen = false;
let prerollFrames = [];

let onStateChange = () => {};
let onPartialTranscript = (_text) => {};
let onFinalTranscript = (_text) => {};
let onBotPartial = (_text) => {};
let onBotResponse = (_text) => {};
let onBotTts = (_text) => {};
let onTurnStarted = (_payload) => {};

function setVoiceState(state, payload = {}) {
  currentVoiceState = state;
  onStateChange(state, payload);
}

function resetDetector() {
  speechFrameStreak = 0;
  silenceFrameStreak = 0;
  isDetectorOpen = false;
  prerollFrames = [];
  adaptiveNoiseFloor = VAD_THRESHOLD * 0.55;
}

function clearPlaybackTimer() {
  if (playbackStateTimer) {
    window.clearTimeout(playbackStateTimer);
    playbackStateTimer = null;
  }
}

function clearBrowserRecognitionRestartTimer() {
  if (browserRecognitionRestartTimer) {
    window.clearTimeout(browserRecognitionRestartTimer);
    browserRecognitionRestartTimer = null;
  }
}

function clearBrowserCommitTimers() {
  if (browserSttCommitTimer) {
    window.clearTimeout(browserSttCommitTimer);
    browserSttCommitTimer = null;
  }
  if (browserSttIdleTimer) {
    window.clearTimeout(browserSttIdleTimer);
    browserSttIdleTimer = null;
  }
}

function clearBrowserResumeTimer() {
  if (browserSttResumeTimer) {
    window.clearTimeout(browserSttResumeTimer);
    browserSttResumeTimer = null;
  }
}

function clearBrowserWatchdogTimer() {
  if (browserSttWatchdogTimer) {
    window.clearTimeout(browserSttWatchdogTimer);
    browserSttWatchdogTimer = null;
  }
}

function stopPlayback() {
  clearPlaybackTimer();
  audioQueue = [];
  isPlayingQueue = false;

  if (currentAudioSource) {
    try {
      currentAudioSource.stop();
      currentAudioSource.disconnect();
    } catch (error) {
      console.debug("Playback stop ignored:", error);
    }
    currentAudioSource = null;
  }
}

function stopBrowserSpeech() {
  pendingListeningPayload = null;
  browserTtsQueue = [];
  browserTtsActive = false;
  browserTtsCurrent = null;
  clearBrowserResumeTimer();

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

function supportsBrowserVoiceStt() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

function supportsBrowserVoiceTts() {
  return !!("speechSynthesis" in window && "SpeechSynthesisUtterance" in window);
}

function resolveVoiceModes() {
  activeSttMode = STT_MODE === "browser" && supportsBrowserVoiceStt() ? "browser" : "server";
  activeTtsMode = TTS_MODE === "browser" && supportsBrowserVoiceTts() ? "browser" : "server";
}

function sendVoiceControl(payload) {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return false;
  }

  voiceSocket.send(JSON.stringify(payload));
  return true;
}

function getBrowserTranscriptSnapshot() {
  return `${browserFinalTranscript} ${browserLatestInterim}`.replace(/\s+/g, " ").trim();
}

function resetBrowserTranscriptState() {
  clearBrowserCommitTimers();
  browserFinalTranscript = "";
  browserLatestInterim = "";
}

function disposeBrowserRecognition() {
  if (!browserRecognition) {
    return;
  }

  try {
    browserRecognition.onstart = null;
    browserRecognition.onresult = null;
    browserRecognition.onerror = null;
    browserRecognition.onend = null;
    browserRecognition.stop();
  } catch (error) {
    console.debug("Browser recognition dispose ignored:", error);
  }

  browserRecognition = null;
  browserRecognitionRunning = false;
}

function stopBrowserRecognition({ keepIntent = false } = {}) {
  clearBrowserRecognitionRestartTimer();
  clearBrowserResumeTimer();
  clearBrowserWatchdogTimer();
  clearBrowserCommitTimers();
  browserRecognitionShouldRun = keepIntent;

  if (browserRecognition) {
    try {
      browserRecognition.stop();
    } catch (error) {
      console.debug("Browser recognition stop ignored:", error);
    }
  }
}

function scheduleBrowserRecognitionWatchdog() {
  clearBrowserWatchdogTimer();
  if (!isVoiceModeActive || activeSttMode !== "browser" || !browserRecognitionShouldRun) {
    return;
  }

  browserSttWatchdogTimer = window.setTimeout(() => {
    browserSttWatchdogTimer = null;
    if (
      isVoiceModeActive &&
      activeSttMode === "browser" &&
      browserRecognitionShouldRun &&
      currentVoiceState === "listening" &&
      !browserRecognitionRunning
    ) {
      restartBrowserRecognition();
      return;
    }
    scheduleBrowserRecognitionWatchdog();
  }, BROWSER_STT_WATCHDOG_MS);
}

function restartBrowserRecognition() {
  if (activeSttMode !== "browser" || !supportsBrowserVoiceStt()) {
    return;
  }

  disposeBrowserRecognition();
  browserRecognitionShouldRun = true;
  ensureBrowserRecognition();
}

function commitBrowserTranscript(reason = "final") {
  const transcript = getBrowserTranscriptSnapshot();
  resetBrowserTranscriptState();

  if (!transcript || currentVoiceState === "thinking" || currentVoiceState === "speaking") {
    return;
  }

  browserRecognitionShouldRun = false;
  stopBrowserRecognition({ keepIntent: false });
  onFinalTranscript({ text: transcript, mode: "browser", reason });
  sendVoiceControl({ type: "voice_text_final", text: transcript });
}

function scheduleBrowserCommit(delayMs, reason) {
  if (!supportsBrowserVoiceStt()) {
    return;
  }

  if (browserSttCommitTimer) {
    window.clearTimeout(browserSttCommitTimer);
  }

  browserSttCommitTimer = window.setTimeout(() => {
    browserSttCommitTimer = null;
    commitBrowserTranscript(reason);
  }, delayMs);
}

function scheduleBrowserIdleCommit() {
  if (browserSttIdleTimer) {
    window.clearTimeout(browserSttIdleTimer);
  }

  browserSttIdleTimer = window.setTimeout(() => {
    browserSttIdleTimer = null;
    commitBrowserTranscript("idle");
  }, BROWSER_STT_IDLE_COMMIT_MS);
}

function ensureBrowserRecognition() {
  if (
    activeSttMode !== "browser" ||
    !supportsBrowserVoiceStt() ||
    browserRecognitionRunning ||
    !browserRecognitionShouldRun ||
    currentVoiceState === "thinking" ||
    currentVoiceState === "speaking"
  ) {
    return;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!browserRecognition) {
    browserRecognition = new SpeechRecognition();
    browserRecognition.lang = "en-IN";
    browserRecognition.interimResults = true;
    browserRecognition.continuous = true;

    browserRecognition.onstart = () => {
      browserRecognitionRunning = true;
      scheduleBrowserRecognitionWatchdog();
    };

    browserRecognition.onresult = (event) => {
      let sawFinal = false;
      let interim = "";

      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const transcript = String(result[0]?.transcript || "").trim();
        if (!transcript) {
          continue;
        }

        if (result.isFinal) {
          browserFinalTranscript = `${browserFinalTranscript} ${transcript}`.trim();
          sawFinal = true;
        } else {
          interim += ` ${transcript}`;
        }
      }

      browserLatestInterim = interim.trim();
      const snapshot = getBrowserTranscriptSnapshot();
      if (snapshot) {
        onPartialTranscript({ text: snapshot, mode: "browser", final: sawFinal });
      }

      if (sawFinal) {
        scheduleBrowserCommit(BROWSER_STT_COMMIT_DEBOUNCE_MS, "final");
      } else if (snapshot) {
        scheduleBrowserIdleCommit();
      }
    };

    browserRecognition.onerror = (event) => {
      console.warn("Live browser STT error:", event);
      browserRecognitionRunning = false;
      if (event?.error === "not-allowed" || event?.error === "service-not-allowed") {
        browserRecognitionShouldRun = false;
        clearBrowserWatchdogTimer();
      } else if (browserRecognitionShouldRun && isVoiceModeActive) {
        clearBrowserRecognitionRestartTimer();
        browserRecognitionRestartTimer = window.setTimeout(() => {
          browserRecognitionRestartTimer = null;
          restartBrowserRecognition();
        }, 180);
      }
    };

    browserRecognition.onend = () => {
      browserRecognitionRunning = false;
      scheduleBrowserRecognitionWatchdog();
      if (
        browserRecognitionShouldRun &&
        isVoiceModeActive &&
        currentVoiceState !== "thinking" &&
        currentVoiceState !== "speaking"
      ) {
        clearBrowserRecognitionRestartTimer();
        browserRecognitionRestartTimer = window.setTimeout(() => {
          browserRecognitionRestartTimer = null;
          restartBrowserRecognition();
        }, 140);
      }
    };
  }

  try {
    browserRecognition.start();
    scheduleBrowserRecognitionWatchdog();
  } catch (error) {
    console.debug("Browser recognition start ignored:", error);
  }
}

function trySetListeningAfterBrowserSpeech() {
  if (!isVoiceModeActive || browserTtsActive || browserTtsQueue.length) {
    return;
  }

  const payload = pendingListeningPayload || { state: "listening" };
  pendingListeningPayload = null;
  if (activeSttMode === "browser") {
    browserRecognitionShouldRun = true;
    resetBrowserTranscriptState();
  }
  setVoiceState("listening", payload);
  if (activeSttMode === "browser") {
    clearBrowserResumeTimer();
    browserSttResumeTimer = window.setTimeout(() => {
      browserSttResumeTimer = null;
      restartBrowserRecognition();
    }, BROWSER_STT_RESUME_AFTER_TTS_MS);
  }
}

function speakNextBrowserChunk() {
  if (activeTtsMode !== "browser" || !supportsBrowserVoiceTts()) {
    return;
  }

  if (browserTtsActive || !browserTtsQueue.length) {
    trySetListeningAfterBrowserSpeech();
    return;
  }

  const text = browserTtsQueue.shift();
  if (!text) {
    trySetListeningAfterBrowserSpeech();
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.02;
  utterance.pitch = 1.0;
  browserTtsActive = true;
  browserTtsCurrent = utterance;
  if (activeSttMode === "browser") {
    resetBrowserTranscriptState();
    stopBrowserRecognition({ keepIntent: false });
  }
  setVoiceState("speaking", { text });
  utterance.onstart = () => {};
  utterance.onend = () => {
    if (browserTtsCurrent === utterance) {
      browserTtsCurrent = null;
      browserTtsActive = false;
      speakNextBrowserChunk();
    }
  };
  utterance.onerror = () => {
    if (browserTtsCurrent === utterance) {
      browserTtsCurrent = null;
      browserTtsActive = false;
      speakNextBrowserChunk();
    }
  };

  browserTtsCurrent = utterance;
  window.speechSynthesis.speak(utterance);
}

function enqueueBrowserSpeech(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned || activeTtsMode !== "browser" || !supportsBrowserVoiceTts()) {
    return;
  }

  browserTtsQueue.push(cleaned);
  speakNextBrowserChunk();
}

function analyzeInputFrame(inputData) {
  let sumSquares = 0;
  let peak = 0;
  let zeroCrossings = 0;
  let absSum = 0;
  let prev = inputData[0] || 0;

  for (let i = 0; i < inputData.length; i += 1) {
    const sample = inputData[i];
    const absSample = Math.abs(sample);
    sumSquares += sample * sample;
    absSum += absSample;
    peak = Math.max(peak, absSample);

    if ((sample >= 0 && prev < 0) || (sample < 0 && prev >= 0)) {
      zeroCrossings += 1;
    }
    prev = sample;
  }

  return {
    rms: Math.sqrt(sumSquares / inputData.length),
    peak,
    avgAbs: absSum / inputData.length,
    zcr: zeroCrossings / inputData.length,
  };
}

function updateNoiseFloor(rms) {
  if (!Number.isFinite(rms) || rms <= 0) {
    return;
  }

  if (rms < adaptiveNoiseFloor * 1.55 || adaptiveNoiseFloor < VAD_THRESHOLD * 0.8) {
    adaptiveNoiseFloor = Math.max(VAD_THRESHOLD * 0.22, (adaptiveNoiseFloor * 0.9) + (rms * 0.1));
  }
}

function floatToPcmBuffer(inputData) {
  const pcmData = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i += 1) {
    pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
  }
  return pcmData.buffer;
}

function rememberPrerollFrame(buffer) {
  if (currentVoiceState === "speaking" || currentVoiceState === "thinking") {
    return;
  }
  prerollFrames.push(buffer);
  if (prerollFrames.length > VAD_MAX_PREROLL_FRAMES) {
    prerollFrames.shift();
  }
}

function flushPrerollFrames() {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    prerollFrames = [];
    return;
  }

  prerollFrames.forEach((frame) => {
    voiceSocket.send(frame);
  });
  prerollFrames = [];
}

function playPCM16(audioCtx, pcmBuffer, onEnded) {
  const int16 = new Int16Array(pcmBuffer);
  const float32 = new Float32Array(int16.length);

  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }

  const buffer = audioCtx.createBuffer(1, float32.length, AUDIO_SAMPLE_RATE);
  buffer.getChannelData(0).set(float32);

  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  currentAudioSource = src;

  src.onended = () => {
    if (currentAudioSource === src) {
      currentAudioSource = null;
      onEnded?.();
    }
  };

  src.start();
}

function playNextAudioChunk() {
  if (!audioContext || currentAudioSource || !audioQueue.length) {
    if (!currentAudioSource && !audioQueue.length) {
      isPlayingQueue = false;
      if (isVoiceModeActive && currentVoiceState === "speaking") {
        setVoiceState("listening");
      }
    }
    return;
  }

  isPlayingQueue = true;
  const chunk = audioQueue.shift();
  playPCM16(audioContext, chunk, () => {
    playNextAudioChunk();
  });
}

function sendInterruptSignal() {
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return;
  }

  stopPlayback();
  stopBrowserSpeech();
  if (activeSttMode === "browser") {
    resetBrowserTranscriptState();
    browserRecognitionShouldRun = true;
  }
  voiceSocket.send(JSON.stringify({ type: "interrupt" }));
  if (activeSttMode === "browser") {
    setVoiceState("listening", { state: "listening", reason: "local-interrupt" });
    clearBrowserResumeTimer();
    browserSttResumeTimer = window.setTimeout(() => {
      browserSttResumeTimer = null;
      restartBrowserRecognition();
    }, 120);
  }
}

function handleSocketMessage(event) {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
      case "partial":
        onPartialTranscript({ text: msg.text || "", turnId: msg.turnId ?? null, mode: "server" });
        break;

      case "final":
        onFinalTranscript({ text: msg.text || "", turnId: msg.turnId ?? null, mode: "server" });
        break;

      case "bot_response":
        onBotResponse({ text: msg.text || "", turnId: msg.turnId ?? null });
        break;

      case "bot_partial":
        onBotPartial({ text: msg.text || "", turnId: msg.turnId ?? null });
        break;

      case "bot_tts":
        onBotTts({ text: msg.text || "", turnId: msg.turnId ?? null });
        break;

      case "turn_started":
        activeAssistantTurnId = msg.turnId ?? null;
        onTurnStarted({ turnId: msg.turnId ?? null, text: msg.text || "" });
        break;

      case "status":
        if (msg.state === "interrupted") {
          stopPlayback();
          stopBrowserSpeech();
          if (activeSttMode === "browser") {
            browserRecognitionShouldRun = true;
            ensureBrowserRecognition();
          }
          setVoiceState("listening", msg);
          break;
        }

        if (msg.state === "listening" && activeTtsMode === "browser" && (browserTtsActive || browserTtsQueue.length)) {
          pendingListeningPayload = msg;
          break;
        }

        setVoiceState(msg.state || "idle", msg);
        if (activeSttMode === "browser") {
          if (msg.state === "listening") {
            browserRecognitionShouldRun = true;
            restartBrowserRecognition();
          } else if (msg.state === "thinking" || msg.state === "speaking") {
            stopBrowserRecognition({ keepIntent: false });
          }
        }
        break;
    }

    return;
  }

  if (!(event.data instanceof ArrayBuffer) || !audioContext) {
    return;
  }

  clearPlaybackTimer();
  resetDetector();
  setVoiceState("speaking");
  audioQueue.push(event.data);
  playNextAudioChunk();

  const approxDurationMs = Math.max(400, Math.round((event.data.byteLength / 2 / AUDIO_SAMPLE_RATE) * 1000));
  playbackStateTimer = window.setTimeout(() => {
    if (isVoiceModeActive && currentVoiceState === "speaking" && !currentAudioSource && !audioQueue.length) {
      setVoiceState("listening");
    }
  }, approxDurationMs + 120);
}

function setupWebSocket() {
  voiceSocket = new WebSocket(WEBSOCKET_URL);
  voiceSocket.binaryType = "arraybuffer";

  voiceSocket.onopen = () => {
    voiceSocket.send(JSON.stringify({ type: "session_config", ttsMode: activeTtsMode, sttMode: activeSttMode }));
    setVoiceState("connecting");
  };

  voiceSocket.onmessage = handleSocketMessage;

  voiceSocket.onclose = () => {
    if (isVoiceModeActive) {
      stopVoiceMode();
    }
  };

  voiceSocket.onerror = (error) => {
    console.error("VoiceSocket error:", error);
    stopVoiceMode("error", { detail: "Voice connection failed." });
  };
}

function processAudio(event) {
  if (activeSttMode === "browser") {
    return;
  }

  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Accept user audio whenever the session is active, but keep it blocked while the assistant is speaking or thinking.
  if (currentVoiceState === "speaking" || currentVoiceState === "thinking") {
    resetDetector();
    return;
  }

  const inputData = event.inputBuffer.getChannelData(0);
  const stats = analyzeInputFrame(inputData);
  const pcmBuffer = floatToPcmBuffer(inputData);
  const dynamicThreshold = Math.max(VAD_THRESHOLD, adaptiveNoiseFloor * VAD_NOISE_MULTIPLIER);
  const relaxedThreshold = Math.max(VAD_THRESHOLD * 0.78, adaptiveNoiseFloor * (VAD_NOISE_MULTIPLIER - 0.3));
  const plausibleVoiceBand = stats.zcr >= VAD_MIN_ZCR && stats.zcr <= VAD_MAX_ZCR;
  const peakRatio = stats.rms > 0 ? stats.peak / stats.rms : 0;
  const keyboardLike = peakRatio >= KEYBOARD_PEAK_RATIO || stats.zcr > KEYBOARD_MAX_ZCR;
  const nearFieldLike =
    stats.peak >= dynamicThreshold * NEAR_FIELD_PEAK_MULTIPLIER &&
    stats.rms >= adaptiveNoiseFloor * NEAR_FIELD_RMS_MULTIPLIER;
  const speechLike =
    stats.rms >= dynamicThreshold &&
    plausibleVoiceBand &&
    stats.peak >= dynamicThreshold * 1.45 &&
    !keyboardLike &&
    nearFieldLike;
  const keepAlive =
    stats.rms >= relaxedThreshold &&
    plausibleVoiceBand &&
    stats.peak >= relaxedThreshold * 1.2 &&
    !keyboardLike;

  if (stats.rms >= adaptiveNoiseFloor * 0.7) {
    rememberPrerollFrame(pcmBuffer);
  }

  if (!isDetectorOpen) {
    if (!speechLike) {
      speechFrameStreak = 0;
      updateNoiseFloor(stats.rms);
      return;
    }

    speechFrameStreak += 1;
    if (speechFrameStreak < VAD_TRIGGER_FRAMES) {
      return;
    }

    isDetectorOpen = true;
    silenceFrameStreak = 0;
    flushPrerollFrames();
    return;
  }

  if (speechLike || keepAlive) {
    silenceFrameStreak = 0;
    voiceSocket.send(pcmBuffer);
    return;
  }

  silenceFrameStreak += 1;
  if (silenceFrameStreak <= VAD_HANGOVER_FRAMES) {
    voiceSocket.send(pcmBuffer);
    return;
  }

  updateNoiseFloor(stats.rms);
  speechFrameStreak = 0;
  silenceFrameStreak = 0;
  isDetectorOpen = false;
}

async function startVoiceMode() {
  if (isVoiceModeActive) {
    return;
  }

  isVoiceModeActive = true;
  setVoiceState("connecting");

  try {
    activeAssistantTurnId = null;
    resolveVoiceModes();
    stopBrowserSpeech();
    resetDetector();
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: AUDIO_SAMPLE_RATE,
    });
    setupWebSocket();
    if (activeSttMode === "browser") {
      browserRecognitionShouldRun = true;
      return;
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    processorNode.onaudioprocess = processAudio;

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);
  } catch (error) {
    console.error("Failed to start voice mode:", error);
    stopVoiceMode("error", { detail: "Microphone access failed." });
  }
}

function stopVoiceMode(finalState = "idle", payload = {}) {
  if (!isVoiceModeActive) {
    return;
  }

  isVoiceModeActive = false;
  activeAssistantTurnId = null;
  stopPlayback();
  stopBrowserSpeech();
  browserRecognitionShouldRun = false;
  stopBrowserRecognition({ keepIntent: false });
  disposeBrowserRecognition();
  resetBrowserTranscriptState();
  resetDetector();

  if (voiceSocket) {
    const socket = voiceSocket;
    voiceSocket = null;
    socket.close();
  }

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  setVoiceState(finalState, payload);
}

export function initVoiceMode(callbacks) {
  onStateChange = callbacks.onStateChange;
  onPartialTranscript = callbacks.onPartialTranscript;
  onFinalTranscript = callbacks.onFinalTranscript;
  onBotPartial = callbacks.onBotPartial || (() => {});
  onBotResponse = callbacks.onBotResponse;
  onBotTts = callbacks.onBotTts || ((payload) => enqueueBrowserSpeech(payload?.text || payload));
  onTurnStarted = callbacks.onTurnStarted || (() => {});

  return {
    start: startVoiceMode,
    stop: stopVoiceMode,
    interrupt: sendInterruptSignal,
    isActive: () => isVoiceModeActive,
  };
}
