//voice.socket.js
const WebSocket = require("ws");
const handleVoiceSession = require("../services/sts/realtimests");

module.exports = function initVoiceSocket(server) {
  const wss = new WebSocket.Server({ server, path: "/ws/voice" });

  wss.on("connection", (ws) => {
    handleVoiceSession(ws);
  });
};
