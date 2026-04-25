const axios = require("axios");

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getNowParts(timeZone) {
  const now = new Date();

  return {
    time: new Intl.DateTimeFormat("en-IN", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
      timeZone,
    }).format(now),
    date: new Intl.DateTimeFormat("en-IN", {
      day: "numeric",
      month: "long",
      year: "numeric",
      timeZone,
    }).format(now),
    weekday: new Intl.DateTimeFormat("en-IN", {
      weekday: "long",
      timeZone,
    }).format(now),
  };
}

function getRealtimeReply(message, timeZone) {
  const text = normalizeText(message);
  const now = getNowParts(timeZone);

  const asksTime =
    /\b(what is|what s|tell me|current|right now)?\s*(the )?time\b/.test(text) ||
    /\btime now\b/.test(text);
  const asksDate =
    /\b(what is|what s|tell me|current|today s|todays)?\s*(the )?date\b/.test(text) ||
    /\btoday s date\b/.test(text) ||
    /\bdate today\b/.test(text);
  const asksDay =
    /\bwhat day is it\b/.test(text) ||
    /\bwhich day is it\b/.test(text) ||
    /\bwhat day is today\b/.test(text) ||
    /\bwhich day is today\b/.test(text) ||
    /\bcurrent day\b/.test(text) ||
    /\bday today\b/.test(text);

  if (asksTime && asksDate) {
    return `Right now it's ${now.time} on ${now.weekday}, ${now.date}.`;
  }

  if (asksTime) {
    return `Right now it's ${now.time}.`;
  }

  if (asksDate && asksDay) {
    return `Today is ${now.weekday}, ${now.date}.`;
  }

  if (asksDate) {
    return `Today's date is ${now.date}.`;
  }

  if (asksDay) {
    return `Today is ${now.weekday}.`;
  }

  return null;
}

class LLMProvider {
  constructor() {
    this.chatbotApiUrl =
      process.env.AI_CHATBOT_API_URL || "http://127.0.0.1:5010/api/chat";
    this.localUrl = process.env.LOCAL_LLM_URL || null;
    this.timeoutMs = Number(process.env.LLM_TIMEOUT_MS || 60000);
    this.timeZone = process.env.APP_TIMEZONE || Intl.DateTimeFormat().resolvedOptions().timeZone;
  }

  async generateReply(message, history = [], userId = "voice-default") {
    const realtimeReply = getRealtimeReply(message, this.timeZone);
    if (realtimeReply) {
      return realtimeReply;
    }

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

