// stt/webspeech.stt.js
let rec = null;
let recognizing = false;
let finalTranscript = "";

export function isSupported() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

export function start({ onPartial, onStatus }) {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  rec = new SpeechRecognition();
  rec.lang = "en-IN";
  rec.interimResults = true;
  rec.continuous = true;
  finalTranscript = "";

  rec.onstart = () => {
    recognizing = true;
    onStatus?.("recording");
  };

  rec.onresult = (ev) => {
    let interim = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const r = ev.results[i];
      if (r.isFinal) {
        finalTranscript += r[0].transcript + " ";
      } else {
        interim += r[0].transcript;
      }
    }
    onPartial((finalTranscript + interim).trim());
  };

  rec.onerror = (e) => {
    console.warn("WebSpeech error", e);
    stop();
  };

  rec.onend = () => {
    if (recognizing) {
      setTimeout(() => {
        try { rec.start(); } catch { }
      }, 200);
    } else {
      onStatus?.("ended");
    }
  };

  rec.start();
}

export function stop() {
  recognizing = false;
  if (rec) {
    try { rec.stop(); } catch { }
    rec = null;
  }
}
