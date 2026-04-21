//azure.sts.js
const sdk = require("microsoft-cognitiveservices-speech-sdk");

class AzureSTS {
  constructor() {
    this.speechConfig = sdk.SpeechConfig.fromSubscription(
      process.env.AZURE_SPEECH_KEY,
      process.env.AZURE_SPEECH_REGION
    );

    this.speechConfig.speechSynthesisVoiceName =
      "en-IN-NeerjaNeural";

    // Node SDK exposes this as a property on SpeechConfig (some versions don't expose the setter)
    this.speechConfig.speechSynthesisOutputFormat =
      sdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm;
  }

  createRecognizer(pushStream) {
    const audioConfig = sdk.AudioConfig.fromStreamInput(pushStream);
    return new sdk.SpeechRecognizer(this.speechConfig, audioConfig);
  }

  createSynthesizer() {
    return new sdk.SpeechSynthesizer(this.speechConfig);
  }
}

module.exports = AzureSTS;
