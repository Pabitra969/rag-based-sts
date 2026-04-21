// services/speech/index.js
const AzureSpeechProvider = require("./azure.provider");
const OllamaSpeechProvider = require("./ollama.provider");

function getSpeechProvider() {
  const provider = process.env.SPEECH_PROVIDER;

  switch (provider) {
    case "azure":
      return new AzureSpeechProvider();
    case "ollama":
      return new OllamaSpeechProvider();
    default:
      throw new Error("Invalid SPEECH_PROVIDER");
  }
}

module.exports = getSpeechProvider;
