//server.js
const express = require("express");
const cors = require("cors");
const http = require("http");
require("dotenv").config();

const speechTokenRouter = require("./routes/speechToken.route");
const initVoiceSocket = require("./routes/voice.socket");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/speech-token", speechTokenRouter);

const port = process.env.PORT || 5005;

// HTTP server (single source of truth)
const server = http.createServer(app);

// Attach WebSocket
initVoiceSocket(server);

// ONE listen only
server.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
