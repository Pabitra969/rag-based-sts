// services/speech/ollama.provider.js
const SpeechProvider = require("./speech.interface");

class OllamaSpeechProvider extends SpeechProvider {
  async getToken() {
    // example – local STT usually doesn't need token
    return {
      token: null,
      region: "local",
    };
  }
}

module.exports = OllamaSpeechProvider;
