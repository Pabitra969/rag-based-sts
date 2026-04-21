// stt/azure.stt.js
let recognizer = null;
let finalText = "";

async function getToken() {
  const res = await fetch("http://localhost:5005/api/speech-token");
  return res.json();
}

export async function start({ onPartial, onFinal, onStatus }) {
  // Reset finalText for new session
  finalText = "";
  
  const { token, region } = await getToken();

  const speechConfig =
    SpeechSDK.SpeechConfig.fromAuthorizationToken(token, region);
  speechConfig.speechRecognitionLanguage = "en-IN";

  const audioConfig =
    SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();

  recognizer = new SpeechSDK.SpeechRecognizer(
    speechConfig,
    audioConfig
  );

  recognizer.recognizing = (_, e) => {
    if (!e.result.text) return;
    const currentText = finalText
      ? finalText + " " + e.result.text
      : e.result.text;
    onPartial?.(currentText);
  };

  recognizer.recognized = (_, e) => {
    if (!e.result.text) return;
    finalText += e.result.text + " ";
    const completeText = finalText.trim();
    onPartial?.(completeText);
    onFinal?.(completeText);
  };

  recognizer.sessionStopped = () => {
    stop();
    onStatus?.("ended");
  };
  
  recognizer.canceled = (_, e) => {
    console.error("Azure Speech recognition canceled:", e.errorDetails);
    stop();
    onStatus?.("error");
  };

  recognizer.startContinuousRecognitionAsync(
    () => {
      onStatus?.("recording");
    },
    (err) => {
      console.error("Failed to start Azure Speech recognition:", err);
      onStatus?.("error");
    }
  );
}

export function stop() {
  if (!recognizer) return;
  recognizer.stopContinuousRecognitionAsync(() => {
    recognizer.close();
    recognizer = null;
    finalText = ""; // Reset final text when stopping
  });
}
