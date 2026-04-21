// stt/local.stt.js
let mediaRecorder = null;
let audioChunks = [];
let stream = null;

export async function start({ onPartial, onStatus }) {
  try {
    // 1️⃣ Ask mic permission
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // 2️⃣ Create MediaRecorder
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
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      await sendAudioToBackend(audioBlob, onPartial);
    };

    mediaRecorder.start();

  } catch (err) {
    console.error("Mic permission denied or error", err);
    alert("Microphone access is required.");
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

// 🔁 send audio to backend
async function sendAudioToBackend(blob, onPartial) {
  try {
    const formData = new FormData();
    formData.append("audio", blob, "speech.webm");

    const res = await fetch("http://localhost:5000/api/stt/local", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    // Expected backend response:
    // { text: "recognized speech" }
    if (data?.text) {
      onPartial?.(data.text);
    }

  } catch (err) {
    console.error("Failed to send audio", err);
  }
}
