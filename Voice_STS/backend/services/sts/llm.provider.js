const axios = require("axios");
class LLMProvider {
  constructor() {
    this.chatbotApiUrl =
      process.env.AI_CHATBOT_API_URL || "http://127.0.0.1:5010/api/chat";
    this.localUrl = process.env.LOCAL_LLM_URL || null;
    this.timeoutMs = Number(process.env.LLM_TIMEOUT_MS || 60000);
  }

  async generateReply(message, history = [], userId = "voice-default") {
    // Primary path: AI_CHATBOT FastAPI endpoint
    if (this.chatbotApiUrl) {
      try {
        const chatbotRes = await axios.post(
          this.chatbotApiUrl,
          {
            user_id: userId,
            query: message,
          },
          { timeout: this.timeoutMs }
        );

        const chatbotText =
          chatbotRes.data?.answer ||
          chatbotRes.data?.text ||
          chatbotRes.data?.message ||
          "";

        if (String(chatbotText).trim()) {
          return String(chatbotText).trim();
        }
      } catch (err) {
        console.error("AI_CHATBOT call failed:", err.message);
      }
    }

    // Fallback path: generic local LLM endpoint
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
        timeout: this.timeoutMs,
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

