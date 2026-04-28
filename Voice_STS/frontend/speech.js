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
    hint: "Speak naturally. Your voice is accepted only while listening is active.",
  },
  thinking: {
    label: "Thinking.....",
    hint: "Your last utterance is locked in. New voice/chat input is paused until reply is ready.",
  },
  speaking: {
    label: "Speaking.....",
    hint: "Assistant audio is playing. New voice/chat input is paused until playback completes.",
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
let lastVoiceFinalAt = 0;
let lastVoiceBotText = "";
let lastVoiceBotAt = 0;
let lastVoiceBotTurnId = null;
let currentVoiceAssistantState = "idle";
let renderedConversationId = null;
let textTypingIndicatorEl = null;
let isTextReplyPending = false;
let liveVoiceBotDraft = null;
let activeVoiceAssistantTurnId = null;

function parseProductLine(line) {
  const trimmed = String(line || "").trim().replace(/^-+\s*/, "");
  const pipeMatch = trimmed.match(/^(.*?)\s*\|\s*(₹[\d,]+)\s*\|\s*([^.]+)\.\s*(.+)$/);
  const dashMatch = trimmed.match(/^(.*?)\s*\|\s*(₹[\d,]+)\s*\|\s*([^.]+)\s*[-:]\s*(.+)$/);
  const legacyMatch = trimmed.match(/^(.*?)\s*(₹[\d,]+)\s*\[([^\]]+)\]\.\s*(.+)$/);
  const match = pipeMatch || dashMatch || legacyMatch;

  if (!match) {
    return null;
  }

  return {
    title: match[1].trim(),
    price: match[2].trim(),
    category: match[3].trim(),
    description: match[4].trim(),
  };
}

function extractProductCards(text) {
  const lines = String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const cards = [];
  const intro = [];

  lines.forEach((line) => {
    const card = parseProductLine(line);
    if (card) {
      cards.push(card);
    } else {
      intro.push(line);
    }
  });

  if (!cards.length) {
    return null;
  }

  return {
    intro: intro.join("\n"),
    cards,
  };
}

function setTextGenerationLock(isLocked) {
  isTextReplyPending = isLocked;

  // Voice mode controls its own lock state; avoid overriding that mode.
  if (uiState === UI_STATE.VOICE_CHAT) {
    return;
  }

  textInput.disabled = isLocked;
  micTranscribeBtn.disabled = isLocked;
  sendBtn.disabled = isLocked;
  voiceChatBtn.disabled = isLocked;
}

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
      conversations = [...stored].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
      activeConversationId = conversations[0].id;
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
    activeConversation = [...conversations].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))[0] || createConversation();

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
      if (voiceMode?.isActive()) {
        endVoiceSession({ closePanel: true });
      }
      resetVoiceUiState();
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
    chatBody.appendChild(createMessageBubble(message).bubble);
  });

  if (liveVoiceBotDraft?.conversationId === activeConversation.id && liveVoiceBotDraft.text) {
    const streamingBubble = createVoiceDraftBubble(liveVoiceBotDraft.text);
    chatBody.appendChild(streamingBubble.bubble);
  }

  scrollChatToBottom();
}

function createMessageBubble(message) {
  const bubble = document.createElement("div");
  bubble.className = `message ${message.sender}`;
  const productView = message.sender === "bot" ? extractProductCards(message.text) : null;
  let text = null;

  if (productView) {
    if (productView.intro) {
      text = document.createElement("p");
      text.className = "message-text";
      text.textContent = productView.intro;
      bubble.appendChild(text);
    }

    const productList = document.createElement("div");
    productList.className = "product-list";

    productView.cards.forEach((product) => {
      const card = document.createElement("article");
      card.className = "product-card";

      const top = document.createElement("div");
      top.className = "product-card-top";

      const title = document.createElement("h3");
      title.className = "product-title";
      title.textContent = product.title;

      const price = document.createElement("span");
      price.className = "product-price";
      price.textContent = product.price;

      top.appendChild(title);
      top.appendChild(price);

      const category = document.createElement("div");
      category.className = "product-category";
      category.textContent = product.category;

      const description = document.createElement("p");
      description.className = "product-description";
      description.textContent = product.description;

      card.appendChild(top);
      card.appendChild(category);
      card.appendChild(description);
      productList.appendChild(card);
    });

    bubble.appendChild(productList);
  } else {
    text = document.createElement("p");
    text.className = "message-text";
    text.textContent = message.text;
    bubble.appendChild(text);
  }

  const meta = document.createElement("p");
  meta.className = "message-meta";
  meta.textContent = formatTimestamp(message.createdAt);

  bubble.appendChild(meta);
  return { bubble, text, meta };
}

function animateTextContent(target, text, stepMs = 50) {
  const words = String(text || "").split(/(\s+)/).filter(Boolean);
  target.textContent = "";

  if (!words.length) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    let index = 0;
    const timer = window.setInterval(() => {
      target.textContent += words[index];
      index += 1;
      scrollChatToBottom();

      if (index >= words.length) {
        window.clearInterval(timer);
        resolve();
      }
    }, stepMs);
  });
}

function createTypingIndicator() {
  const bubble = document.createElement("div");
  bubble.className = "message bot typing-indicator";

  const dots = document.createElement("div");
  dots.className = "typing-dots";
  dots.innerHTML = "<span></span><span></span><span></span>";

  bubble.appendChild(dots);
  return bubble;
}

function createStreamingBubble() {
  const bubble = document.createElement("div");
  bubble.className = "message bot";

  const text = document.createElement("p");
  text.className = "message-text";
  text.textContent = "";

  const meta = document.createElement("p");
  meta.className = "message-meta";
  meta.textContent = formatTimestamp(Date.now());

  bubble.appendChild(text);
  bubble.appendChild(meta);
  return { bubble, text };
}

function createVoiceDraftBubble(textValue = "") {
  const streamingBubble = createStreamingBubble();
  streamingBubble.bubble.classList.add("voice-live-draft");
  streamingBubble.text.textContent = textValue;
  return streamingBubble;
}

function showTextTypingIndicator() {
  if (textTypingIndicatorEl) {
    return;
  }

  chatBody.querySelector(".empty-state")?.remove();
  textTypingIndicatorEl = createTypingIndicator();
  chatBody.appendChild(textTypingIndicatorEl);
  scrollChatToBottom();
}

function removeTextTypingIndicator() {
  if (!textTypingIndicatorEl) {
    return;
  }

  textTypingIndicatorEl.remove();
  textTypingIndicatorEl = null;
}

function updateLiveVoiceDraft(text, conversationId = voiceConversationId || activeConversationId, turnId = null) {
  const draftText = String(text || "").trim();
  if (!draftText || !conversationId) {
    return;
  }

  if (turnId && liveVoiceBotDraft?.turnId && liveVoiceBotDraft.turnId !== turnId) {
    return;
  }

  liveVoiceBotDraft = {
    conversationId,
    turnId: turnId ?? liveVoiceBotDraft?.turnId ?? activeVoiceAssistantTurnId ?? null,
    text: draftText,
  };

  if (renderedConversationId === conversationId) {
    const lastBubble = chatBody.querySelector(".voice-live-draft:last-of-type");
    if (lastBubble) {
      const textEl = lastBubble.querySelector(".message-text");
      if (textEl) {
        textEl.textContent = draftText;
        scrollChatToBottom();
        return;
      }
    }
  }

  renderApp();
}

function clearLiveVoiceDraft(conversationId = null, turnId = null) {
  if (!liveVoiceBotDraft) {
    return;
  }

  if (conversationId && liveVoiceBotDraft.conversationId !== conversationId) {
    return;
  }

  if (turnId && liveVoiceBotDraft.turnId !== turnId) {
    return;
  }

  liveVoiceBotDraft = null;
  renderApp();
}

function startLiveVoiceDraft(turnId, seedText = "", conversationId = voiceConversationId || activeConversationId) {
  if (!conversationId || !turnId) {
    return;
  }

  activeVoiceAssistantTurnId = turnId;
  liveVoiceBotDraft = {
    conversationId,
    turnId,
    text: String(seedText || "").trim(),
  };
  renderApp();
}

function scrollChatToBottom() {
  requestAnimationFrame(() => {
    chatBody.scrollTop = chatBody.scrollHeight;
    chatBody.lastElementChild?.scrollIntoView({ block: "end" });
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
      sendBtn.classList.remove("hidden");
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

function addMessageToConversation(sender, text, conversationId = activeConversationId, options = {}) {
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
      const { bubble, text: textEl } = createMessageBubble(message);
      chatBody.appendChild(bubble);
      scrollChatToBottom();
      renderConversationList();
      updateVoiceAssistant(currentVoiceAssistantState);
      if (options.animate && sender === "bot" && textEl) {
        textEl.textContent = "";
        void animateTextContent(textEl, message.text, options.stepMs || 20);
      }
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

function stopTranscriptionSession({ updateUi = true } = {}) {
  if (!sttProvider) {
    if (updateUi) {
      uiState = textInput.value.trim() ? UI_STATE.TYPING : UI_STATE.IDLE;
      renderUI();
    }
    return;
  }

  try {
    sttProvider.stop?.();
  } catch (error) {
    console.warn("Failed to stop transcription provider:", error);
  }

  sttProvider = null;

  if (updateUi) {
    uiState = textInput.value.trim() ? UI_STATE.TYPING : UI_STATE.IDLE;
    renderUI();
    autosizeInput();
    if (textInput.value.trim()) {
      textInput.focus();
    }
  }
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
  if (isTextReplyPending || uiState === UI_STATE.VOICE_CHAT) {
    return;
  }

  if (uiState === UI_STATE.TRANSCRIBING) {
    stopTranscriptionSession({ updateUi: true });
  }

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
  setTextGenerationLock(true);
  chatBody.querySelector(".empty-state")?.remove();
  const streamingBubble = createStreamingBubble();
  chatBody.appendChild(streamingBubble.bubble);
  scrollChatToBottom();

  try {
    const response = await fetch(`${CHAT_API_URL}/stream`, {
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

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Streaming is not supported by this browser.");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        const line = event
          .split("\n")
          .find((entry) => entry.startsWith("data: "));
        if (!line) {
          continue;
        }

        const payload = JSON.parse(line.slice(6));
        if (payload.type === "delta") {
          answer += payload.text || "";
          streamingBubble.text.textContent = answer;
          scrollChatToBottom();
        } else if (payload.type === "error") {
          throw new Error(payload.error || "stream failed");
        }
      }
    }

    streamingBubble.bubble.remove();
    removeTextTypingIndicator();
    answer = answer.trim();
    addMessageToConversation("bot", answer || "Sorry, no response received.", targetConversationId, {
      animate: false,
    });
  } catch (error) {
    console.error("Text chat failed:", error);
    streamingBubble.bubble.remove();
    addMessageToConversation(
      "bot",
      "Could not reach chatbot backend. Make sure Voice_STS backend is running on 127.0.0.1:5005 and AI_CHATBOT is running on 127.0.0.1:5010.",
      targetConversationId,
      { animate: true, stepMs: 14 }
    );
  } finally {
    setTextGenerationLock(false);
  }
}

function beginVoiceSession() {
  openVoiceAssistantPanel();
  voiceConversationId = activeConversationId;
  lastVoiceFinalText = "";
  lastVoiceBotText = "";
  lastVoiceBotTurnId = null;
  liveVoiceBotDraft = null;
  activeVoiceAssistantTurnId = null;
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

function resetVoiceUiState() {
  lastVoiceFinalText = "";
  lastVoiceFinalAt = 0;
  lastVoiceBotText = "";
  lastVoiceBotAt = 0;
  lastVoiceBotTurnId = null;
  liveVoiceBotDraft = null;
  activeVoiceAssistantTurnId = null;
  voiceConversationId = null;
  voiceUserTranscript.textContent = "Waiting for your voice…";
  voiceAssistantTranscript.textContent = "Responses will appear here and in the main chat.";
}

const voiceModeCallbacks = {
  onStateChange: (state, payload = {}) => {
    if (state === "idle") {
      clearLiveVoiceDraft(voiceConversationId, payload.turnId || null);
      activeVoiceAssistantTurnId = null;
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
      clearLiveVoiceDraft(voiceConversationId, payload.turnId || null);
      activeVoiceAssistantTurnId = null;
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
    if (state === "interrupted" && payload.turnId) {
      clearLiveVoiceDraft(voiceConversationId || activeConversationId, payload.turnId);
    }
    updateVoiceAssistant(state, payload);
  },

  onPartialTranscript: (payload) => {
    const partial = String(payload?.text || payload || "").trim();
    if (!partial) {
      return;
    }

    voiceUserTranscript.textContent = partial;
    textInput.value = partial;
    autosizeInput();
  },

  onFinalTranscript: (payload) => {
    const finalText = String(payload?.text || payload || "").trim();
    if (!finalText) {
      return;
    }

    if (finalText === lastVoiceFinalText && Date.now() - lastVoiceFinalAt < 2600) {
      return;
    }

    lastVoiceFinalText = finalText;
    lastVoiceFinalAt = Date.now();
    voiceUserTranscript.textContent = finalText;
    addMessageToConversation("user", finalText, voiceConversationId || activeConversationId);
    textInput.value = "";
    autosizeInput();
  },

  onTurnStarted: ({ turnId, text }) => {
    if (!turnId) {
      return;
    }

    startLiveVoiceDraft(turnId, text, voiceConversationId || activeConversationId);
  },

  onBotPartial: (payload) => {
    const partial = String(payload?.text || payload || "").trim();
    const turnId = payload?.turnId ?? null;
    if (!partial) {
      return;
    }

    if (turnId) {
      activeVoiceAssistantTurnId = turnId;
      if (!liveVoiceBotDraft || liveVoiceBotDraft.turnId !== turnId) {
        startLiveVoiceDraft(turnId, "", voiceConversationId || activeConversationId);
      }
    }

    voiceAssistantTranscript.textContent = partial;
    updateLiveVoiceDraft(partial, voiceConversationId || activeConversationId, turnId);
  },

  onBotResponse: (payload) => {
    const reply = String(payload?.text || payload || "").trim();
    const turnId = payload?.turnId ?? null;
    if (!reply) {
      return;
    }

    if ((turnId && turnId === lastVoiceBotTurnId) || (!turnId && reply === lastVoiceBotText && Date.now() - lastVoiceBotAt < 1200)) {
      return;
    }

    if (turnId && liveVoiceBotDraft?.turnId && liveVoiceBotDraft.turnId !== turnId) {
      return;
    }

    lastVoiceBotText = reply;
    lastVoiceBotAt = Date.now();
    lastVoiceBotTurnId = turnId;
    activeVoiceAssistantTurnId = turnId;
    voiceAssistantTranscript.textContent = reply;
    clearLiveVoiceDraft(voiceConversationId || activeConversationId, turnId);
    addMessageToConversation("bot", reply, voiceConversationId || activeConversationId, {
      animate: false,
    });
  },
};

document.addEventListener("DOMContentLoaded", () => {
  loadConversations();
  voiceMode = initVoiceMode(voiceModeCallbacks);
  renderApp();

  newChatBtn.addEventListener("click", () => {
    if (voiceMode?.isActive()) {
      endVoiceSession({ closePanel: true });
    }
    resetVoiceUiState();
    ensureActiveConversationHasRoom();
    textInput.value = "";
    autosizeInput();
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
    if (isTextReplyPending) {
      return;
    }

    if (uiState === UI_STATE.TRANSCRIBING) {
      stopTranscriptionSession({ updateUi: true });
      return;
    }

    uiState = UI_STATE.TRANSCRIBING;
    renderUI();

    sttProvider = getSTTProvider();
    Promise.resolve(sttProvider.start({
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
          stopTranscriptionSession({ updateUi: true });
        }
      },
    })).catch((error) => {
      console.error("Failed to start transcription:", error);
      stopTranscriptionSession({ updateUi: true });
    });
  });

  voiceChatBtn.addEventListener("click", () => {
    if (isTextReplyPending) {
      return;
    }

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
