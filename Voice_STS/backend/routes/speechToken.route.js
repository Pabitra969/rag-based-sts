// routes/speechToken.route.js
const express = require("express");
const rateLimit = require("express-rate-limit");
const getSpeechProvider = require("../services/speech");

const router = express.Router();

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
});

router.get("/", limiter, async (req, res) => {
  try {
    const speechProvider = getSpeechProvider();
    const data = await speechProvider.getToken();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
