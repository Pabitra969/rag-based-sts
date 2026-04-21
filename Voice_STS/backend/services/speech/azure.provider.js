// services/speech/azure.provider.js
const axios = require("axios");
const SpeechProvider = require("./speech.interface");

class AzureSpeechProvider extends SpeechProvider {
  async getToken() {
    const response = await axios.post(
      `https://${process.env.AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken`,
      null,
      {
        headers: {
          "Ocp-Apim-Subscription-Key": process.env.AZURE_SPEECH_KEY,
        },
      }
    );

    return {
      token: response.data,
      region: process.env.AZURE_SPEECH_REGION,
    };
  }
}

module.exports = AzureSpeechProvider;
