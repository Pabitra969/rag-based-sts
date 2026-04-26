const express = require("express");
const axios = require("axios");

const router = express.Router();

const audioBody = express.raw({
  type: ["audio/webm", "audio/wav", "audio/mpeg", "application/octet-stream"],
  limit: "25mb",
});

function getLocalSttUrl() {
  return String(process.env.LOCAL_STT_URL || "").trim();
}

router.get("/health", async (_req, res) => {
  const localSttUrl = getLocalSttUrl();
  const healthUrl = String(process.env.LOCAL_STT_HEALTH_URL || "").trim();

  if (!localSttUrl) {
    return res.status(503).json({
      available: false,
      error: "LOCAL_STT_URL is not configured",
    });
  }

  if (!healthUrl) {
    return res.json({ available: true });
  }

  try {
    await axios.get(healthUrl, { timeout: 1500 });
    return res.json({ available: true });
  } catch (err) {
    return res.status(503).json({
      available: false,
      error: err.message,
    });
  }
});

router.post("/", audioBody, async (req, res) => {
  const localSttUrl = getLocalSttUrl();

  if (!localSttUrl) {
    return res.status(503).json({
      error: "LOCAL_STT_URL is not configured",
    });
  }

  if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
    return res.status(400).json({ error: "audio body is required" });
  }

  try {
    const upstream = await axios.post(localSttUrl, req.body, {
      headers: {
        "Content-Type": req.headers["content-type"] || "audio/webm",
      },
      timeout: Number(process.env.LOCAL_STT_TIMEOUT_MS || 30000),
    });

    const text =
      upstream.data?.text ||
      upstream.data?.transcript ||
      upstream.data?.result ||
      "";

    return res.json({ text: String(text).trim() });
  } catch (err) {
    console.error("Local STT proxy failed:", err.message);
    return res.status(502).json({ error: "local STT failed" });
  }
});

module.exports = router;
