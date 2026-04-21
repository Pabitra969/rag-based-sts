const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(sender, text) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.textContent = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    appendMessage("user", text);
    userInput.value = "";

    appendMessage("bot", "Typing...");

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: "local_user", query: text })
        });

        const data = await res.json();
        chatBox.lastChild.remove(); // remove "Thinking..." message

        if (data.answer) {
            appendMessage("bot", data.answer);
        } else if (data.error) {
            appendMessage("bot", "⚠️ Error: " + data.error);
        }
    } catch (err) {
        chatBox.lastChild.remove();
        appendMessage("bot", "❌ Network error: " + err.message);
    }
}

sendBtn.onclick = sendMessage;
userInput.addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
});
