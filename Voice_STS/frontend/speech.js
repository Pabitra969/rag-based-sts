// // frontend/speech.js
// import { getSTTProvider } from "./stt/index.js";
// import { initVoiceMode } from "./voice/voiceMode.js";

// // --- DOM Elements ---
// const textInput = document.getElementById("textInput");
// const micTranscribeBtn = document.getElementById("micTranscribeBtn");
// const voiceChatBtn = document.getElementById("voiceChatBtn");
// const sendBtn = document.getElementById("sendBtn");
// const chatBody = document.getElementById("chatBody");

// // --- State ---
// let sttProvider = null;
// let voiceMode = null;

// // --- Button Toggle Helper ---
// function updateButtonVisibility() {
//     // Don't change buttons during recording or voice mode
//     if (sttProvider || voiceMode?.isActive()) {
//         return;
//     }

//     const isFocused = document.activeElement === textInput;
//     const hasText = textInput.value.trim().length > 0;

//     // Show send button when: textfield is focused OR there's text
//     // Show mic button when: textfield is NOT focused AND there's no text
//     if (isFocused || hasText) {
//         sendBtn.classList.remove("hidden");
//         micTranscribeBtn.classList.add("hidden");
//     } else {
//         sendBtn.classList.add("hidden");
//         micTranscribeBtn.classList.remove("hidden");
//     }
// }

// // --- UI Helpers ---
// function addMessage(text, sender) {
//     const div = document.createElement("div");
//     div.className = `message ${sender}`;
//     div.textContent = text;
//     chatBody.appendChild(div);
//     chatBody.scrollTop = chatBody.scrollHeight;
// }

// function setVoiceChatStatus(status) {
//     // Simple visual indicator, can be expanded (e.g., icons, text)
//     voiceChatBtn.dataset.status = status;
//     switch (status) {
//         case "listening":
//             voiceChatBtn.textContent = "🎧";
//             break;
//         case "speaking":
//             voiceChatBtn.textContent = "💬";
//             break;
//         case "connecting":
//             voiceChatBtn.textContent = "🔌";
//             break;
//         case "error":
//             voiceChatBtn.textContent = "⚠️";
//             break;
//         default:
//             voiceChatBtn.textContent = "🎧";
//             break;
//     }
// }


// // --- Voice Mode (Full Duplex) Callbacks ---
// const voiceModeCallbacks = {
//     onStateChange: (state) => {
//         setVoiceChatStatus(state);
//         if (state === "error" || state === "idle") {
//             // Re-enable text input when voice mode stops or fails
//             textInput.disabled = false;
//             micTranscribeBtn.disabled = false;
//             voiceChatBtn.classList.remove("active");
//             // Restore button state based on current input state
//             updateButtonVisibility();
//         } else {
//             // Disable text input during voice mode
//             textInput.disabled = true;
//             micTranscribeBtn.disabled = true;
//             voiceChatBtn.classList.add("active");
//         }
//     },
//     onPartialTranscript: (text) => {
//         textInput.value = text; // Show real-time transcription in input box
//         // Voice mode manages its own UI, keep buttons as is during voice mode
//     },
//     onFinalTranscript: (text) => {
//         addMessage(text, "user");
//         textInput.value = ""; // Clear input after final transcript
//         // After final transcript in voice mode, show mic button
//         if (document.activeElement !== textInput) {
//             sendBtn.classList.add("hidden");
//             micTranscribeBtn.classList.remove("hidden");
//         }
//     },
//     onBotResponse: (text) => {
//         addMessage(text, "bot");
//     },
// };

// // --- Initialization ---
// document.addEventListener("DOMContentLoaded", () => {
//     voiceMode = initVoiceMode(voiceModeCallbacks);

//     // Initialize button state: show mic button, hide send button
//     updateButtonVisibility();

//     // Handle clicks outside to update button visibility
//     document.addEventListener("click", (e) => {
//         // Small delay to let focus/blur events fire first
//         setTimeout(() => {
//             updateButtonVisibility();
//         }, 0);
//     });

//     // --- Event Listeners ---

//     // 1. Text Input Events - Toggle between mic and send button
//     textInput.addEventListener("focus", () => {
//         updateButtonVisibility();
//     });

//     textInput.addEventListener("blur", () => {
//         // Small delay to let other events settle
//         setTimeout(() => {
//             updateButtonVisibility();
//         }, 0);
//     });

//     textInput.addEventListener("input", () => {
//         updateButtonVisibility();
//     });

//     sendBtn.addEventListener("click", () => {
//         const text = textInput.value.trim();
//         if (text) {
//             addMessage(text, "user");
//             textInput.value = "";
//             // Dummy bot reply for text messages
//             setTimeout(() => addMessage("I am a text-only bot.", "bot"), 500);
//         }
//         // Update button visibility after sending
//         setTimeout(() => {
//             updateButtonVisibility();
//         }, 0);
//     });


//     // 2. Real-time Transcription Button
//     micTranscribeBtn.addEventListener("click", () => {
//         if (voiceMode.isActive()) return; // Should be disabled, but as a safeguard

//         if (sttProvider) {
//             // Stop recording
//             sttProvider.stop();
//             sttProvider = null;
//             micTranscribeBtn.classList.remove("active");
//             micTranscribeBtn.textContent = "🎤";
//             // After stopping, update button visibility
//             setTimeout(() => {
//                 updateButtonVisibility();
//             }, 0);
//         } else {
//             // Start recording - show mic button with stop indicator
//             sttProvider = getSTTProviderWithUI();
//             sttProvider.start({
//                 onPartial: (text) => {
//                     textInput.value = text;
//                     // Keep mic button visible while recording (with stop indicator)
//                     micTranscribeBtn.classList.remove("hidden");
//                     sendBtn.classList.add("hidden");
//                 },
//                 onFinal: (text) => {
//                     textInput.value = text;
//                     // After final transcript, update button visibility
//                     setTimeout(() => {
//                         updateButtonVisibility();
//                     }, 0);
//                 },
//                 onStatus: (status) => {
//                     if (status === "ended" || status === "error") {
//                         sttProvider = null;
//                         micTranscribeBtn.classList.remove("active");
//                         micTranscribeBtn.textContent = "🎤";
//                         // Update button visibility after transcription ends
//                         setTimeout(() => {
//                             updateButtonVisibility();
//                         }, 0);
//                     } else if (status === "recording") {
//                         micTranscribeBtn.classList.add("active");
//                         micTranscribeBtn.textContent = "⏹️";
//                     }
//                 }
//             });
//             // Update button state immediately - show mic with stop indicator
//             micTranscribeBtn.classList.add("active");
//             micTranscribeBtn.textContent = "⏹️";
//             micTranscribeBtn.classList.remove("hidden");
//             sendBtn.classList.add("hidden");
//         }
//     });


//     // 3. Full Voice Chat Button
//     voiceChatBtn.addEventListener("click", () => {
//         if (sttProvider) return; // Should be disabled, but as a safeguard

//         if (voiceMode.isActive()) {
//             voiceMode.stop();
//         } else {
//             voiceMode.start();
//         }
//     });
// });


// // Override sttProvider start to attach a status handler that shows send button on end/error
// const originalGetProvider = getSTTProvider;
// function getSTTProviderWithUI() {
//     const provider = originalGetProvider();
//     const originalStart = provider.start;
//     provider.start = (opts = {}) => {
//         const wrappedOpts = {
//             ...opts,
//             onFinal: (text) => {
//                 opts.onFinal?.(text);
//                 setTimeout(() => {
//                     updateButtonVisibility();
//                 }, 0);
//             },
//             onStatus: (status) => {
//                 opts.onStatus?.(status);
//                 if (status === "ended" || status === "error") {
//                     setTimeout(() => {
//                         updateButtonVisibility();
//                     }, 0);
//                 }
//             },
//         };
//         originalStart.call(provider, wrappedOpts);
//     };
//     return provider;
// }




// frontend/speech.js
import { getSTTProvider } from "./stt/index.js";
import { initVoiceMode } from "./voice/voiceMode.js";

/* ---------------- DOM ---------------- */
const textInput = document.getElementById("textInput");
const micTranscribeBtn = document.getElementById("micTranscribeBtn");
const voiceChatBtn = document.getElementById("voiceChatBtn");
const sendBtn = document.getElementById("sendBtn");
const chatBody = document.getElementById("chatBody");
sendBtn.classList.add("hidden");
micTranscribeBtn.classList.remove("hidden");


/* ---------------- UI STATE ---------------- */
const UI_STATE = {
  IDLE: "idle",              // no text, no recording
  TYPING: "typing",          // user typing / text present
  TRANSCRIBING: "transcribing", // push-to-talk STT
  VOICE_CHAT: "voice_chat"   // full duplex voice mode
};

let uiState = UI_STATE.IDLE;
let sttProvider = null;
let voiceMode = null;

/* ---------------- UI RENDER ---------------- */
function renderUI() {
  // default
  sendBtn.classList.add("hidden");
  micTranscribeBtn.classList.add("hidden");

  switch (uiState) {
    case UI_STATE.IDLE:
      micTranscribeBtn.classList.remove("hidden");
      break;

    case UI_STATE.TYPING:
      sendBtn.classList.remove("hidden");
      break;

    case UI_STATE.TRANSCRIBING:
      micTranscribeBtn.classList.remove("hidden");
      micTranscribeBtn.textContent = "⏹️";
      micTranscribeBtn.classList.add("active");
      break;

    case UI_STATE.VOICE_CHAT:
      // buttons stay disabled / unchanged
      break;
  }
}

/* ---------------- CHAT HELPERS ---------------- */
function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

/* ---------------- VOICE MODE ---------------- */
const voiceModeCallbacks = {
  onStateChange: (state) => {
    if (state === "idle" || state === "error") {
      uiState = UI_STATE.IDLE;
      textInput.disabled = false;
      micTranscribeBtn.disabled = false;
      voiceChatBtn.classList.remove("active");
      renderUI();
    } else {
      uiState = UI_STATE.VOICE_CHAT;
      textInput.disabled = true;
      micTranscribeBtn.disabled = true;
      voiceChatBtn.classList.add("active");
    }
  },

  onPartialTranscript: (text) => {
    textInput.value = text;
  },

  onFinalTranscript: (text) => {
    addMessage(text, "user");
    textInput.value = "";
    uiState = UI_STATE.IDLE;
    renderUI();
  },

  onBotResponse: (text) => {
    addMessage(text, "bot");
  }
};

/* ---------------- INIT ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  voiceMode = initVoiceMode(voiceModeCallbacks);
  renderUI();

  /* -------- TEXT INPUT -------- */
  function updateInputState() {
    if (uiState === UI_STATE.TRANSCRIBING || uiState === UI_STATE.VOICE_CHAT) return;

    const hasText = textInput.value.trim().length > 0;
    const isFocused = document.activeElement === textInput;

    // Strict toggle: If Focused OR Has Text -> TYPING (Send Btn). Else -> IDLE (Mic Btn).
    uiState = (hasText || isFocused) ? UI_STATE.TYPING : UI_STATE.IDLE;
    renderUI();
  }

  textInput.addEventListener("input", updateInputState);
  textInput.addEventListener("focus", updateInputState);
  textInput.addEventListener("blur", () => {
    // Small delay to allow click events on buttons to register before hiding
    setTimeout(updateInputState, 100);
  });

  /* -------- SEND -------- */
  sendBtn.addEventListener("click", () => {
    const text = textInput.value.trim();
    if (!text) return;

    addMessage(text, "user");
    textInput.value = "";
    uiState = UI_STATE.IDLE;
    renderUI();

    setTimeout(() => {
      addMessage("I am a text-only bot.", "bot");
    }, 400);
  });

  /* -------- MIC TRANSCRIBE -------- */
  micTranscribeBtn.addEventListener("click", () => {
    if (uiState === UI_STATE.TRANSCRIBING) {
      if (sttProvider) {
        sttProvider.stop();
      }
      return;
    }

    uiState = UI_STATE.TRANSCRIBING;
    renderUI();

    sttProvider = getSTTProvider();
    sttProvider.start({
      onPartial: (text) => {
        textInput.value = text;
      },
      onFinal: (text) => {
        textInput.value = text;
      },
      onStatus: (status) => {
        if (status === "ended" || status === "error") {
          sttProvider = null;
          micTranscribeBtn.textContent = "🎤";
          micTranscribeBtn.classList.remove("active");

          // Directly check text presence to set state immediately
          const hasText = textInput.value.trim().length > 0;
          if (hasText) {
            uiState = UI_STATE.TYPING;
            // Optionally focus, but ensure state is set first
            textInput.focus();
          } else {
            uiState = UI_STATE.IDLE;
          }
          renderUI();
        }
      }
    });
  });

  /* -------- FULL VOICE CHAT -------- */
  voiceChatBtn.addEventListener("click", () => {
    if (voiceMode.isActive()) {
      voiceMode.stop();
    } else {
      voiceMode.start();
    }
  });
});
