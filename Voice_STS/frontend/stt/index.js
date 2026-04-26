// stt/index.js
import * as webspeech from "./webspeech.stt.js";
import * as azure from "./azure.stt.js";
import * as local from "./local.stt.js";

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
  return {
    async start(callbacks = {}) {
      if (await local.isAvailable()) {
        console.log("✅ Local STT available - Using local STT");
        return local.start(callbacks);
      }

      if (isChromium() && webspeech.isSupported()) {
        console.log("✅ Local STT unavailable - Using WebSpeech API STT");
        return webspeech.start(callbacks);
      }

      console.log("⚠️ Local/WebSpeech unavailable - Using Azure Speech STT backup");
      return azure.start(callbacks);
    },

    stop() {
      local.stop?.();
      webspeech.stop?.();
      azure.stop?.();
    }
  };
}
