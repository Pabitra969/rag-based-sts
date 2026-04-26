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

  async generateReply(message, history = [], userId = "voice-default", options = {}) {
    const { signal } = options;

    // Primary path: AI_CHATBOT FastAPI endpoint
    if (this.chatbotApiUrl) {
      try {
        const chatbotRes = await axios.post(
          this.chatbotApiUrl,
          {
            user_id: userId,
            query: message,
            voice_mode: true,
          },
          { timeout: this.timeoutMs, signal }
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
        if (axios.isCancel?.(err) || err?.code === "ERR_CANCELED") {
          throw err;
        }
        console.error("AI_CHATBOT call failed:", err.message);
      }
    }

    // Fallback path: generic local LLM endpoint
    if (!this.localUrl) {
      return "I couldn't reach the chatbot backend just now. Please check that AI_CHATBOT is running.";
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
        signal,
      });

      const replyText =
        res.data?.text ||
        res.data?.message ||
        "I'm here. How can I help?";

      return String(replyText).trim();
    } catch (err) {
      if (axios.isCancel?.(err) || err?.code === "ERR_CANCELED") {
        throw err;
      }
      console.error("LLMProvider local call failed:", err.message);
      return "Sorry, my brain is offline right now.";
    }
  }

  async *streamReply(message, history = [], userId = "voice-default", options = {}) {
    const { signal } = options;

    if (this.chatbotApiUrl) {
      const chatbotStreamUrl = this.chatbotApiUrl.replace(/\/api\/chat$/, "/api/chat/stream");

      try {
        const chatbotRes = await axios.post(
          chatbotStreamUrl,
          {
            user_id: userId,
            query: message,
            voice_mode: true,
          },
          {
            timeout: this.timeoutMs,
            signal,
            responseType: "stream",
            headers: {
              Accept: "text/event-stream",
            },
          }
        );

        let buffer = "";
        for await (const chunk of chatbotRes.data) {
          buffer += chunk.toString("utf8");
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const event of events) {
            const line = event
              .split("\n")
              .find((entry) => entry.startsWith("data: "));
            if (!line) {
              continue;
            }

            const payload = JSON.parse(line.slice(6));
            if (payload.type === "delta" && payload.text) {
              yield String(payload.text);
            } else if (payload.type === "error") {
              throw new Error(payload.error || "stream failed");
            }
          }
        }
        return;
      } catch (err) {
        if (axios.isCancel?.(err) || err?.code === "ERR_CANCELED") {
          throw err;
        }
        console.error("AI_CHATBOT stream failed:", err.message);
      }
    }

    const reply = await this.generateReply(message, history, userId, options);
    if (String(reply).trim()) {
      yield String(reply).trim();
    }
  }
}

module.exports = LLMProvider;
