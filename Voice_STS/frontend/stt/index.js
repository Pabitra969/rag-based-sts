// stt/index.js
import * as webspeech from "./webspeech.stt.js";
import * as azure from "./azure.stt.js";

/**
 * Detects if the browser is Chromium-based (Chrome, Edge, Chromium, Opera, etc.)
 * Web Speech API is primarily supported in Chromium-based browsers.
 * For non-Chromium browsers (Firefox, Safari), we fall back to Azure Speech.
 */
function isChromium() {
  const ua = navigator.userAgent.toLowerCase();
  // Check for Chromium-based browsers
  // Exclude Safari which has "chrome" in its user agent but is not Chromium
  const isSafari = ua.includes("safari") && !ua.includes("chrome");
  if (isSafari) return false;
  
  return (
    ua.includes("chrome") ||
    ua.includes("edg") ||
    ua.includes("chromium") ||
    ua.includes("opera") ||
    ua.includes("opr")
  );
}

/**
 * Gets the appropriate STT provider based on browser compatibility.
 * - For Chromium-based browsers: Uses Web Speech API (native, no backend needed)
 * - For non-Chromium browsers: Uses Azure Speech SDK (requires backend token)
 */
export function getSTTProvider() {
  if (isChromium() && webspeech.isSupported()) {
    console.log("✅ Chromium browser detected - Using WebSpeech API STT");
    return webspeech;
  }

  console.log("⚠️ Non-Chromium browser or WebSpeech not supported - Using Azure Speech STT");
  return azure;
}
