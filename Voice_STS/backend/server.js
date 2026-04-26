//server.js
const express = require("express");
const cors = require("cors");
const http = require("http");
require("dotenv").config();

const speechTokenRouter = require("./routes/speechToken.route");
const localSttRouter = require("./routes/localStt.route");
const initVoiceSocket = require("./routes/voice.socket");
const LLMProvider = require("./services/sts/llm.provider");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/speech-token", speechTokenRouter);
app.use("/api/stt/local", localSttRouter);

app.post("/api/chat", async (req, res) => {
  const query = String(req.body?.query || "").trim();
  const userId = String(req.body?.user_id || "voice-text-user");

  if (!query) {
    return res.status(400).json({ error: "query is required" });
  }

  try {
    const llm = new LLMProvider();
    const answer = await llm.generateReply(query, [], userId);
    return res.json({ answer });
  } catch (err) {
    console.error("/api/chat error:", err.message);
    return res.status(500).json({ error: "failed to generate reply" });
  }
});

const port = process.env.PORT || 5005;

// HTTP server (single source of truth)
const server = http.createServer(app);

// Attach WebSocket
initVoiceSocket(server);

// ONE listen only
server.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
