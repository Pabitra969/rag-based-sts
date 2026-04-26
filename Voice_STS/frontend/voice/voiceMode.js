const VAD_THRESHOLD = 0.0025;
const AUDIO_SAMPLE_RATE = 16000;
const DEFAULT_WS = "ws://localhost:5005/ws/voice";
const WEBSOCKET_URL = window.VOICE_WS_URL || DEFAULT_WS;

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

let onStateChange = () => {};
let onPartialTranscript = (_text) => {};
let onFinalTranscript = (_text) => {};
let onBotPartial = (_text) => {};
let onBotResponse = (_text) => {};

function setVoiceState(state, payload = {}) {
  currentVoiceState = state;
  onStateChange(state, payload);
}

function clearPlaybackTimer() {
  if (playbackStateTimer) {
    window.clearTimeout(playbackStateTimer);
    playbackStateTimer = null;
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

  voiceSocket.send(JSON.stringify({ type: "interrupt" }));
}

function handleSocketMessage(event) {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
      case "partial":
        onPartialTranscript(msg.text || "");
        break;

      case "final":
        onFinalTranscript(msg.text || "");
        break;

      case "bot_response":
        onBotResponse(msg.text || "");
        break;

      case "bot_partial":
        onBotPartial(msg.text || "");
        break;

      case "status":
        if (msg.state === "interrupted") {
          stopPlayback();
          setVoiceState("listening", msg);
          break;
        }

        setVoiceState(msg.state || "idle", msg);
        break;
    }

    return;
  }

  if (!(event.data instanceof ArrayBuffer) || !audioContext) {
    return;
  }

  clearPlaybackTimer();
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
  if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Only accept user microphone audio while backend marks the session as listening.
  if (currentVoiceState !== "listening") {
    return;
  }

  const inputData = event.inputBuffer.getChannelData(0);

  const pcmData = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
  }

  voiceSocket.send(pcmData.buffer);
}

async function startVoiceMode() {
  if (isVoiceModeActive) {
    return;
  }

  isVoiceModeActive = true;
  setVoiceState("connecting");

  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: AUDIO_SAMPLE_RATE,
    });
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    setupWebSocket();

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
  stopPlayback();

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

  return {
    start: startVoiceMode,
    stop: stopVoiceMode,
    interrupt: sendInterruptSignal,
    isActive: () => isVoiceModeActive,
  };
}
