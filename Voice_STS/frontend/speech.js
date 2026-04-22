import { getSTTProvider } from "./stt/index.js";
import { initVoiceMode } from "./voice/voiceMode.js";

const textInput = document.getElementById("textInput");
const micTranscribeBtn = document.getElementById("micTranscribeBtn");
const voiceChatBtn = document.getElementById("voiceChatBtn");
const sendBtn = document.getElementById("sendBtn");
const chatBody = document.getElementById("chatBody");
const chatTitle = document.getElementById("chatTitle");
const chatWorkspace = document.getElementById("chatWorkspace");
const conversationList = document.getElementById("conversationList");
const newChatBtn = document.getElementById("newChatBtn");
const composerForm = document.getElementById("composerForm");
const voiceAssistantPanel = document.getElementById("voiceAssistantPanel");
const voicePanelCloseBtn = document.getElementById("voicePanelCloseBtn");
const voicePanelTitle = document.getElementById("voicePanelTitle");
const voiceVisualizer = document.getElementById("voiceVisualizer");
const voiceStateLabel = document.getElementById("voiceStateLabel");
const voiceStateHint = document.getElementById("voiceStateHint");
const voiceUserTranscript = document.getElementById("voiceUserTranscript");
const voiceAssistantTranscript = document.getElementById("voiceAssistantTranscript");

const CHAT_API_URL = window.TEXT_CHAT_API_URL || "http://127.0.0.1:5005/api/chat";
const STORAGE_KEY = "voice_sts_conversations_v1";

const UI_STATE = {
  IDLE: "idle",
  TYPING: "typing",
  TRANSCRIBING: "transcribing",
  VOICE_CHAT: "voice_chat",
};

const VOICE_COPY = {
  idle: {
    label: "Idle",
    hint: "Open voice chat to start a live back-and-forth session.",
  },
  connecting: {
    label: "Connecting.....",
    hint: "Preparing microphone and live speech channel...",
  },
  listening: {
    label: "Listening.....",
    hint: "Speak naturally. You can interrupt the assistant any time.",
  },
  thinking: {
    label: "Thinking.....",
    hint: "Your last utterance is locked in. The assistant is preparing a reply.",
  },
  speaking: {
    label: "Speaking.....",
    hint: "Assistant audio is playing. Start talking to barge in and take the turn.",
  },
  interrupted: {
    label: "Interrupted",
    hint: "Assistant playback was stopped. Keep talking with your next request.",
  },
  error: {
    label: "Voice error",
    hint: "The live voice session hit a problem. Check the backend and microphone access.",
  },
};

let uiState = UI_STATE.IDLE;
let sttProvider = null;
let voiceMode = null;
let conversations = [];
let activeConversationId = null;
let voiceConversationId = null;
let isVoicePanelOpen = false;
let lastVoiceFinalText = "";
let lastVoiceBotText = "";
let currentVoiceAssistantState = "idle";
let renderedConversationId = null;

function createConversation(title = "New conversation") {
  return {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    title,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
}

function loadConversations() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(stored) && stored.length > 0) {
      conversations = stored;
      activeConversationId = stored[0].id;
      return;
    }
  } catch (error) {
    console.warn("Failed to load conversation history:", error);
  }

  const initialConversation = createConversation();
  conversations = [initialConversation];
  activeConversationId = initialConversation.id;
  persistConversations();
}

function persistConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function getActiveConversation() {
  let activeConversation = conversations.find((conversation) => conversation.id === activeConversationId);

  if (!activeConversation) {
    activeConversation = conversations[0] || createConversation();

    if (!conversations.length) {
      conversations = [activeConversation];
    }

    activeConversationId = activeConversation.id;
    persistConversations();
  }

  return activeConversation;
}

function getConversationPreview(conversation) {
  const lastMessage = conversation.messages[conversation.messages.length - 1];

  if (!lastMessage) {
    return "No messages yet";
  }

  return lastMessage.text.length > 52
    ? `${lastMessage.text.slice(0, 52)}...`
    : lastMessage.text;
}

function getConversationTitle(conversation) {
  const firstUserMessage = conversation.messages.find((message) => message.sender === "user");

  if (!firstUserMessage) {
    return conversation.title || "New conversation";
  }

  const trimmed = firstUserMessage.text.replace(/\s+/g, " ").trim();
  return trimmed.length > 28 ? `${trimmed.slice(0, 28)}...` : trimmed;
}

function formatTimestamp(value) {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function autosizeInput() {
  textInput.style.height = "auto";
  textInput.style.height = `${Math.min(textInput.scrollHeight, 168)}px`;
}

function openVoiceAssistantPanel() {
  isVoicePanelOpen = true;
  voiceAssistantPanel.classList.remove("hidden");
  chatWorkspace.classList.add("with-voice-panel");
}

function closeVoiceAssistantPanel() {
  isVoicePanelOpen = false;
  voiceAssistantPanel.classList.add("hidden");
  chatWorkspace.classList.remove("with-voice-panel");
}

function updateVoiceAssistant(state, payload = {}) {
  const copy = VOICE_COPY[state] || VOICE_COPY.idle;
  const voiceConversation = conversations.find((conversation) => conversation.id === voiceConversationId) || getActiveConversation();

  currentVoiceAssistantState = state;
  voiceVisualizer.dataset.state = state;
  voiceStateLabel.textContent = copy.label;
  voiceStateHint.textContent = payload.detail || payload.text || copy.hint;
  voicePanelTitle.textContent = voiceConversationId
    ? `Live with ${getConversationTitle(voiceConversation)}`
    : "Assistant live";
}

function renderConversationList() {
  const sortedConversations = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  conversationList.innerHTML = "";

  sortedConversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-card";

    if (conversation.id === activeConversationId) {
      button.classList.add("active");
    }

    button.innerHTML = `
      <p class="conversation-card-title">${escapeHtml(getConversationTitle(conversation))}</p>
      <p class="conversation-card-preview">${escapeHtml(getConversationPreview(conversation))}</p>
      <p class="conversation-card-time">Updated ${formatTimestamp(conversation.updatedAt)}</p>
    `;

    button.addEventListener("click", () => {
      activeConversationId = conversation.id;
      renderApp();
    });

    conversationList.appendChild(button);
  });
}

function renderMessages() {
  const activeConversation = getActiveConversation();
  chatTitle.textContent = getConversationTitle(activeConversation);
  chatBody.innerHTML = "";
  renderedConversationId = activeConversation.id;

  if (!activeConversation.messages.length) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.innerHTML = `
      <h3>Start a fresh conversation</h3>
      <p>Your saved chats appear on the left. Open one anytime and continue where you left off.</p>
    `;
    chatBody.appendChild(emptyState);
    return;
  }

  activeConversation.messages.forEach((message) => {
    chatBody.appendChild(createMessageBubble(message));
  });

  scrollChatToBottom();
}

function createMessageBubble(message) {
  const bubble = document.createElement("div");
  bubble.className = `message ${message.sender}`;

  const text = document.createElement("p");
  text.className = "message-text";
  text.textContent = message.text;

  const meta = document.createElement("p");
  meta.className = "message-meta";
  meta.textContent = formatTimestamp(message.createdAt);

  bubble.appendChild(text);
  bubble.appendChild(meta);
  return bubble;
}

function scrollChatToBottom() {
  requestAnimationFrame(() => {
    chatBody.scrollTop = chatBody.scrollHeight;
  });
}

function renderUI() {
  sendBtn.classList.add("hidden");
  micTranscribeBtn.classList.add("hidden");
  micTranscribeBtn.textContent = "🎤";
  micTranscribeBtn.classList.remove("active");

  switch (uiState) {
    case UI_STATE.IDLE:
      micTranscribeBtn.classList.remove("hidden");
      break;

    case UI_STATE.TYPING:
      sendBtn.classList.remove("hidden");
      break;

    case UI_STATE.TRANSCRIBING:
      micTranscribeBtn.classList.remove("hidden");
      micTranscribeBtn.textContent = "⏹";
      micTranscribeBtn.classList.add("active");
      break;

    case UI_STATE.VOICE_CHAT:
      break;
  }
}

function renderApp() {
  renderConversationList();
  renderMessages();
  renderUI();
  updateVoiceAssistant(currentVoiceAssistantState);
  autosizeInput();
}

function hasRecentDuplicate(conversation, sender, text) {
  const lastMessage = conversation.messages[conversation.messages.length - 1];
  if (!lastMessage) {
    return false;
  }

  return (
    lastMessage.sender === sender &&
    lastMessage.text === text &&
    Date.now() - lastMessage.createdAt < 1500
  );
}

function addMessageToConversation(sender, text, conversationId = activeConversationId) {
  const messageText = String(text || "").trim();
  if (!messageText) {
    return;
  }

  const conversation = conversations.find((item) => item.id === conversationId);
  if (!conversation || hasRecentDuplicate(conversation, sender, messageText)) {
    return;
  }

  conversation.messages.push({
    id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    sender,
    text: messageText,
    createdAt: Date.now(),
  });

  conversation.title = getConversationTitle(conversation);
  conversation.updatedAt = Date.now();
  persistConversations();

  if (conversationId === activeConversationId) {
    if (renderedConversationId === conversationId) {
      const message = conversation.messages[conversation.messages.length - 1];
      chatBody.querySelector(".empty-state")?.remove();
      chatBody.appendChild(createMessageBubble(message));
      scrollChatToBottom();
      renderConversationList();
      updateVoiceAssistant(currentVoiceAssistantState);
    } else {
      renderApp();
    }
  } else {
    renderConversationList();
  }
}

function ensureActiveConversationHasRoom() {
  const activeConversation = getActiveConversation();

  if (activeConversation.messages.length === 0) {
    return activeConversation;
  }

  const nextConversation = createConversation();
  conversations.unshift(nextConversation);
  activeConversationId = nextConversation.id;
  persistConversations();
  renderApp();
  return nextConversation;
}

function setInputStateFromText() {
  if (uiState === UI_STATE.TRANSCRIBING || uiState === UI_STATE.VOICE_CHAT) {
    return;
  }

  uiState = textInput.value.trim() ? UI_STATE.TYPING : UI_STATE.IDLE;
  renderUI();
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function sendTextMessage() {
  const text = textInput.value.trim();
  if (!text) {
    return;
  }

  const conversation = getActiveConversation();
  const targetConversationId = conversation.id;

  addMessageToConversation("user", text, targetConversationId);
  textInput.value = "";
  uiState = UI_STATE.IDLE;
  renderUI();
  autosizeInput();

  try {
    const response = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: targetConversationId,
        query: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const answer = (data.answer || data.text || "").toString().trim();
    addMessageToConversation("bot", answer || "Sorry, no response received.", targetConversationId);
  } catch (error) {
    console.error("Text chat failed:", error);
    addMessageToConversation(
      "bot",
      "Could not reach chatbot backend. Make sure Voice_STS backend is running on 127.0.0.1:5005 and AI_CHATBOT is running on 127.0.0.1:5010.",
      targetConversationId
    );
  }
}

function beginVoiceSession() {
  openVoiceAssistantPanel();
  voiceConversationId = activeConversationId;
  lastVoiceFinalText = "";
  lastVoiceBotText = "";
  voiceUserTranscript.textContent = "Listening for your first utterance…";
  voiceAssistantTranscript.textContent = "Assistant responses will appear here and in the main chat.";
  updateVoiceAssistant("connecting");
  voiceMode.start();
}

function endVoiceSession({ closePanel = false } = {}) {
  voiceMode.stop();

  if (closePanel) {
    closeVoiceAssistantPanel();
  }
}

const voiceModeCallbacks = {
  onStateChange: (state, payload = {}) => {
    if (state === "idle") {
      voiceConversationId = null;
      uiState = textInput.value.trim() ? UI_STATE.TYPING : UI_STATE.IDLE;
      textInput.disabled = false;
      micTranscribeBtn.disabled = false;
      sendBtn.disabled = false;
      voiceChatBtn.classList.remove("active");
      voiceChatBtn.querySelector(".btn-label").textContent = "Voice chat";
      renderUI();
      updateVoiceAssistant("idle", payload);
      autosizeInput();
      return;
    }

    if (state === "error") {
      uiState = textInput.value.trim() ? UI_STATE.TYPING : UI_STATE.IDLE;
      textInput.disabled = false;
      micTranscribeBtn.disabled = false;
      sendBtn.disabled = false;
      voiceChatBtn.classList.remove("active");
      voiceChatBtn.querySelector(".btn-label").textContent = "Voice chat";
      renderUI();
      updateVoiceAssistant("error", payload);
      autosizeInput();
      return;
    }

    uiState = UI_STATE.VOICE_CHAT;
    voiceConversationId = voiceConversationId || activeConversationId;
    textInput.disabled = true;
    micTranscribeBtn.disabled = true;
    sendBtn.disabled = true;
    voiceChatBtn.classList.add("active");
    voiceChatBtn.querySelector(".btn-label").textContent = "Stop voice";
    renderUI();
    updateVoiceAssistant(state, payload);
  },

  onPartialTranscript: (text) => {
    const partial = String(text || "").trim();
    if (!partial) {
      return;
    }

    voiceUserTranscript.textContent = partial;
    textInput.value = partial;
    autosizeInput();
  },

  onFinalTranscript: (text) => {
    const finalText = String(text || "").trim();
    if (!finalText || finalText === lastVoiceFinalText) {
      return;
    }

    lastVoiceFinalText = finalText;
    voiceUserTranscript.textContent = finalText;
    addMessageToConversation("user", finalText, voiceConversationId || activeConversationId);
    textInput.value = "";
    autosizeInput();
  },

  onBotResponse: (text) => {
    const reply = String(text || "").trim();
    if (!reply || reply === lastVoiceBotText) {
      return;
    }

    lastVoiceBotText = reply;
    voiceAssistantTranscript.textContent = reply;
    addMessageToConversation("bot", reply, voiceConversationId || activeConversationId);
  },
};

document.addEventListener("DOMContentLoaded", () => {
  loadConversations();
  voiceMode = initVoiceMode(voiceModeCallbacks);
  renderApp();

  newChatBtn.addEventListener("click", () => {
    ensureActiveConversationHasRoom();
    textInput.focus();
  });

  composerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendTextMessage();
  });

  textInput.addEventListener("input", () => {
    autosizeInput();
    setInputStateFromText();
  });

  textInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await sendTextMessage();
    }
  });

  micTranscribeBtn.addEventListener("click", () => {
    if (uiState === UI_STATE.TRANSCRIBING) {
      sttProvider?.stop();
      return;
    }

    uiState = UI_STATE.TRANSCRIBING;
    renderUI();

    sttProvider = getSTTProvider();
    sttProvider.start({
      onPartial: (text) => {
        textInput.value = text;
        autosizeInput();
      },
      onFinal: (text) => {
        textInput.value = text;
        autosizeInput();
      },
      onStatus: (status) => {
        if (status === "ended" || status === "error") {
          sttProvider = null;
          setInputStateFromText();
          autosizeInput();

          if (textInput.value.trim()) {
            textInput.focus();
          }
        }
      },
    });
  });

  voiceChatBtn.addEventListener("click", () => {
    if (voiceMode.isActive()) {
      endVoiceSession({ closePanel: true });
      return;
    }

    beginVoiceSession();
  });

  voicePanelCloseBtn.addEventListener("click", () => {
    if (voiceMode.isActive()) {
      endVoiceSession({ closePanel: true });
      return;
    }

    closeVoiceAssistantPanel();
  });
});
