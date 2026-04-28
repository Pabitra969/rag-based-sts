// stt/local.stt.js
let mediaRecorder = null;
let audioChunks = [];
let stream = null;
const LOCAL_STT_BASE_URL = "http://localhost:5005/api/stt/local";

export async function isAvailable() {
  try {
    const res = await fetch(`${LOCAL_STT_BASE_URL}/health`, {
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function start({ onPartial, onFinal, onStatus }) {
  if (!(await isAvailable())) {
    throw new Error("Local STT is not available");
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm"
    });

    audioChunks = [];

    mediaRecorder.onstart = () => {
      onStatus?.("recording");
    };

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      try {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const text = await sendAudioToBackend(audioBlob);
        if (text) {
          onPartial?.(text);
          onFinal?.(text);
        }
      } catch (err) {
        console.error("Local STT transcription failed:", err);
        onStatus?.("error");
      }
    };

    mediaRecorder.start();

  } catch (err) {
    console.error("Mic permission denied or error", err);
    onStatus?.("error");
    throw err;
  }
}

export function stop() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
}

async function sendAudioToBackend(blob) {
  const res = await fetch(LOCAL_STT_BASE_URL, {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "audio/webm",
    },
    body: blob
  });

  if (!res.ok) {
    throw new Error(`Local STT failed with HTTP ${res.status}`);
  }

  const data = await res.json();
  return String(data?.text || "").trim();
}
