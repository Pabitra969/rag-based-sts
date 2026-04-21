// frontend/voice/voiceMode.js

// --- Configuration ---
const VAD_THRESHOLD = 0.6; // Voice Activity Detection sensitivity
const VAD_SILENCE_TIMEOUT = 1000; // ms of silence before stopping
const AUDIO_SAMPLE_RATE = 16000; // Required by Azure
const DEFAULT_WS = "ws://localhost:5005/ws/voice";
const WEBSOCKET_URL = window.VOICE_WS_URL || DEFAULT_WS;

// --- State ---
let audioContext;
let mediaStream;
let sourceNode;
let processorNode;
let voiceSocket;
let vadNode;
let vadSilenceStart = null;
let isVoiceModeActive = false;
let isBotSpeaking = false;
let currentAudioSource = null; // Track active audio source for cancellation

// --- UI Callbacks (to be set by main script) ---
let onStateChange = () => { };
let onPartialTranscript = (_text) => { };
let onFinalTranscript = (_text) => { };
let onBotResponse = (_text) => { };

/**
 * Stops the currently playing audio immediately.
 */
function stopPlayback() {
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
            currentAudioSource.disconnect();
        } catch (e) {
            // Ignore errors if already stopped
        }
        currentAudioSource = null;
    }
}

/**
 * Plays raw PCM-16 audio data.
 * @param {AudioContext} audioCtx The AudioContext to use.
 * @param {ArrayBuffer} pcmBuffer Raw PCM data.
 */
function playPCM16(audioCtx, pcmBuffer) {
    const int16 = new Int16Array(pcmBuffer);
    const float32 = new Float32Array(int16.length);

    // Convert 16-bit PCM to 32-bit Float
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
    }

    const buffer = audioCtx.createBuffer(1, float32.length, AUDIO_SAMPLE_RATE);
    buffer.getChannelData(0).set(float32);

    const src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);

    currentAudioSource = src; // Track for cancellation
    src.onended = () => {
        if (currentAudioSource === src) {
            currentAudioSource = null;
        }
    };

    src.start();
}


/**
 * Sets up the WebSocket connection for the voice session.
 */
function setupWebSocket() {
    voiceSocket = new WebSocket(WEBSOCKET_URL);
    voiceSocket.binaryType = "arraybuffer";

    voiceSocket.onopen = () => {
        console.log("🔊 VoiceSocket connected.");
        onStateChange("listening");
    };

    voiceSocket.onmessage = async (event) => {
        if (typeof event.data === "string") {
            const msg = JSON.parse(event.data);
            switch (msg.type) {
                case "partial":
                    onPartialTranscript(msg.text);
                    break;
                case "final":
                    onFinalTranscript(msg.text);
                    break;
                case "bot_response":
                    onBotResponse(msg.text);
                    break;
            }
        } else if (event.data instanceof ArrayBuffer) {
            isBotSpeaking = true;
            onStateChange("speaking");
            playPCM16(audioContext, event.data);
            // A simple way to guess when speaking is done.
            // A more robust solution would involve getting duration from the buffer.
            setTimeout(() => {
                isBotSpeaking = false;
                if (isVoiceModeActive) onStateChange("listening");
            }, 2000); // Approximate duration
        }
    };

    voiceSocket.onclose = () => {
        console.log("🔊 VoiceSocket disconnected.");
        if (isVoiceModeActive) {
            // Unexpected close, try to reconnect or stop.
            stopVoiceMode();
        }
    };

    voiceSocket.onerror = (err) => {
        console.error("🔊 VoiceSocket error:", err);
        stopVoiceMode();
    };
}


/**
 * Main audio processing loop.
 * @param {AudioProcessingEvent} event
 */
function processAudio(event) {
    if (!voiceSocket || voiceSocket.readyState !== WebSocket.OPEN) return;

    const inputData = event.inputBuffer.getChannelData(0);

    // Basic VAD (Voice Activity Detection)
    const energy = inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length;
    if (energy > VAD_THRESHOLD) {
        vadSilenceStart = null; // Reset silence timer on activity
        // If user starts speaking while bot is talking, request interrupt
        if (isBotSpeaking) {
            isBotSpeaking = false;
            stopPlayback(); // <-- Kill local audio immediately
            onStateChange("listening");
            voiceSocket.send(JSON.stringify({ type: "interrupt" }));
        }
    } else if (vadSilenceStart === null) {
        vadSilenceStart = Date.now(); // Start silence timer
    }

    // If silent for too long, stop sending (optional, can be done on backend too)
    if (vadSilenceStart && (Date.now() - vadSilenceStart) > VAD_SILENCE_TIMEOUT) {
        // We could stop here, but for now we'll let the backend handle session end.
    }

    // Convert to 16-bit PCM and send
    const pcmData = new Int16Array(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
    }

    voiceSocket.send(pcmData.buffer);
}


/**
 * Starts the entire voice mode session.
 */
async function startVoiceMode() {
    if (isVoiceModeActive) return;
    console.log("🚀 Starting Voice Mode...");
    isVoiceModeActive = true;
    onStateChange("connecting");

    try {
        // 1. Get Audio Context and Stream
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: AUDIO_SAMPLE_RATE
        });
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });
        sourceNode = audioContext.createMediaStreamSource(mediaStream);

        // 2. Setup WebSocket
        setupWebSocket();

        // 3. Setup ScriptProcessor for audio streaming
        // WARNING: Deprecated, but simpler than AudioWorklet for this context.
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        processorNode.onaudioprocess = processAudio;

        // 4. Connect the audio graph
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination); // Connect to output to avoid garbage collection

    } catch (err) {
        console.error("❌ Failed to start voice mode:", err);
        isVoiceModeActive = false;
        onStateChange("error");
        return;
    }
}

/**
 * Stops the voice mode session and cleans up resources.
 */
function stopVoiceMode() {
    if (!isVoiceModeActive) return;
    console.log("🛑 Stopping Voice Mode...");

    isVoiceModeActive = false;
    onStateChange("idle");

    if (voiceSocket) {
        voiceSocket.close();
        voiceSocket = null;
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
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}


/**
 * Initializes the voice mode module with UI callbacks.
 */
export function initVoiceMode(callbacks) {
    onStateChange = callbacks.onStateChange;
    onPartialTranscript = callbacks.onPartialTranscript;
    onFinalTranscript = callbacks.onFinalTranscript;
    onBotResponse = callbacks.onBotResponse;
    return {
        start: startVoiceMode,
        stop: stopVoiceMode,
        isActive: () => isVoiceModeActive,
    };
}