const sdk = require("microsoft-cognitiveservices-speech-sdk");
const AzureSTS = require("./azure.sts");
const LLMProvider = require("./llm.provider");
const LATEST_TURN_SETTLE_MS = 450;

function safeSend(ws, payload, isBinary = false) {
  if (ws.readyState !== 1) {
    return;
  }

  try {
    ws.send(payload, { binary: isBinary });
  } catch (err) {
    console.error("WebSocket send failed:", err.message);
  }
}

module.exports = function handleVoiceSession(ws) {
  const azure = new AzureSTS();
  const llm = new LLMProvider();
  const sessionUserId = `voice_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const format = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);
  const pushStream = sdk.AudioInputStream.createPushStream(format);
  const recognizer = azure.createRecognizer(pushStream);
  const synthesizer = azure.createSynthesizer();

  const history = [];

  let currentTurn = null;
  let queuedTurn = null;
  let nextTurnId = 0;
  let isClosed = false;
  let queueTimer = null;

  function sendEvent(type, extra = {}) {
    safeSend(ws, JSON.stringify({ type, ...extra }));
  }

  function sendStatus(state, extra = {}) {
    sendEvent("status", { state, ...extra });
  }

  function clearQueueTimer() {
    if (queueTimer) {
      clearTimeout(queueTimer);
      queueTimer = null;
    }
  }

  function scheduleQueueProcessing(delayMs = LATEST_TURN_SETTLE_MS) {
    clearQueueTimer();
    queueTimer = setTimeout(() => {
      queueTimer = null;
      queueMicrotask(processQueue);
    }, delayMs);
  }

  function stopSpeaking() {
    return new Promise((resolve) => {
      synthesizer.stopSpeakingAsync(
        () => resolve(),
        (err) => {
          console.error("stopSpeakingAsync error:", err);
          resolve();
        }
      );
    });
  }

  async function interruptCurrentTurn(reason = "interrupt") {
    if (currentTurn?.abortController) {
      currentTurn.abortController.abort();
    }

    if (currentTurn?.historyInjected) {
      const last = history[history.length - 1];
      if (last?.role === "user" && last.content === currentTurn.text) {
        history.pop();
      }
      currentTurn.historyInjected = false;
    }

    if (currentTurn?.stage === "speaking") {
      await stopSpeaking();
      sendStatus("interrupted", { reason, turnId: currentTurn.id });
    }

    if (currentTurn) {
      currentTurn.cancelled = true;
    }
  }

  function synthesizeReply(turn, replyText) {
    return new Promise((resolve, reject) => {
      synthesizer.speakTextAsync(
        replyText,
        (result) => {
          if (turn.cancelled || currentTurn?.id !== turn.id || isClosed) {
            resolve(false);
            return;
          }

          safeSend(ws, result.audioData, true);
          resolve(true);
        },
        (err) => reject(err)
      );
    });
  }

  async function processQueue() {
    if (currentTurn || !queuedTurn || isClosed) {
      return;
    }

    currentTurn = queuedTurn;
    queuedTurn = null;
    const turn = currentTurn;

    sendStatus("thinking", { turnId: turn.id, text: turn.text });

    const abortController = new AbortController();
    turn.abortController = abortController;
    turn.stage = "thinking";
    turn.historyInjected = true;
    history.push({ role: "user", content: turn.text });

    let replyText = "";

    try {
      replyText = await llm.generateReply(turn.text, history, sessionUserId, {
        signal: abortController.signal,
      });
    } catch (err) {
      if (err?.code !== "ERR_CANCELED") {
        console.error("LLM/TTS error:", err);
        sendEvent("bot_response", {
          text: "Sorry, I ran into a problem thinking about that.",
          turnId: turn.id,
        });
      }
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    if (turn.cancelled || currentTurn?.id !== turn.id || isClosed) {
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    const trimmedReply = String(replyText || "").trim();
    if (!trimmedReply) {
      currentTurn = null;
      scheduleQueueProcessing(0);
      return;
    }

    history.push({ role: "assistant", content: trimmedReply });
    sendEvent("bot_response", { text: trimmedReply, turnId: turn.id });

    turn.stage = "speaking";
    sendStatus("speaking", { turnId: turn.id, text: trimmedReply });

    try {
      await synthesizeReply(turn, trimmedReply);
    } catch (err) {
      if (!turn.cancelled) {
        console.error("Error synthesizing speech:", err);
      }
    }

    if (!turn.cancelled && !isClosed) {
      sendStatus("listening", { turnId: turn.id });
    }

    currentTurn = null;
    scheduleQueueProcessing(0);
  }

  function enqueueTurn(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed || isClosed) {
      return;
    }

    const turn = {
      id: ++nextTurnId,
      text: trimmed,
      cancelled: false,
      stage: "queued",
      abortController: null,
      historyInjected: false,
    };

    sendEvent("final", { text: trimmed, turnId: turn.id });

    if (currentTurn) {
      queuedTurn = turn;
      void interruptCurrentTurn("user-barge-in");
      scheduleQueueProcessing();
      return;
    }

    queuedTurn = turn;
    sendStatus("listening", { reason: "queued-latest", turnId: turn.id });
    scheduleQueueProcessing();
  }

  recognizer.recognizing = (_, e) => {
    if (!e.result.text) {
      return;
    }

    sendEvent("partial", {
      text: e.result.text,
      turnId: currentTurn?.id || queuedTurn?.id || nextTurnId + 1,
    });
  };

  recognizer.recognized = (_, e) => {
    if (!e.result.text) {
      return;
    }

    enqueueTurn(e.result.text);
  };

  recognizer.sessionStarted = () => {
    console.log("STT session started");
    sendStatus("listening");
  };

  recognizer.canceled = (_, e) => {
    console.error("STT canceled:", e.errorDetails);
    sendStatus("error", { detail: e.errorDetails || "Speech recognition stopped." });
  };

  recognizer.startContinuousRecognitionAsync();

  ws.on("message", async (data, isBinary) => {
    if (!isBinary) {
      try {
        const raw = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
        const msg = JSON.parse(raw);

        if (msg.type === "interrupt") {
          await interruptCurrentTurn("frontend-interrupt");
          sendStatus("listening", { reason: "interrupt" });
        }
      } catch (err) {
        console.error("Invalid control message:", err.message);
      }
      return;
    }

    // Ignore inbound user audio while a turn is being generated/spoken.
    if (currentTurn) {
      return;
    }

    pushStream.write(data);
  });

  ws.on("close", async () => {
    isClosed = true;
    clearQueueTimer();
    queuedTurn = null;
    await interruptCurrentTurn("socket-close");
    pushStream.close();
    recognizer.stopContinuousRecognitionAsync();
    synthesizer.close();
  });
};
