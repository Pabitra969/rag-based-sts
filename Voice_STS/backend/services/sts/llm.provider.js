const axios = require("axios");
class LLMProvider {
  constructor() {
    this.localUrl = process.env.LOCAL_LLM_URL || null;
  }

  async generateReply(message, history = []) {
    // No local endpoint configured: use canned reply for demo
    if (!this.localUrl) {
      return "Hey, I am customer Support AI chatbot, How can I help you !";
    }

    try {
      const payload = {
        messages: [
          {
            role: "system",
            content:
              "You are a concise, friendly voice assistant. Keep replies short and natural for speech.",
          },
          ...history,
          { role: "user", content: message },
        ],
      };

      const res = await axios.post(this.localUrl, payload, {
        timeout: 15000,
      });

      const replyText =
        res.data?.text ||
        res.data?.message ||
        "I'm here. How can I help?";

      return String(replyText).trim();
    } catch (err) {
      console.error("LLMProvider local call failed:", err.message);
      return "Sorry, my brain is offline right now.";
    }
  }
}

module.exports = LLMProvider;

