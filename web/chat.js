const API_URL = "/chat";

let chatHistory = [];
let isLoading = false;

async function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value.trim();

  if (!message || isLoading) return;

  // 환영 메시지 숨기기
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();

  // 사용자 메시지 추가
  appendMessage("user", message);
  addToHistory(message);

  input.value = "";
  autoResize(input);

  // 로딩 표시
  setLoading(true);
  const typingId = showTyping();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    removeTyping(typingId);
    appendMessage("bot", data.response);
  } catch (err) {
    removeTyping(typingId);
    appendMessage("bot", "오류가 발생했습니다. 다시 시도해주세요.");
  } finally {
    setLoading(false);
  }
}

function appendMessage(role, text) {
  const container = document.getElementById("chatMessages");

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  if (role === "bot") {
    const icon = document.createElement("div");
    icon.className = "bot-icon";
    icon.textContent = "🤖";
    wrapper.appendChild(icon);
  }

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  container.appendChild(wrapper);
  scrollToBottom();
}

function showTyping() {
  const container = document.getElementById("chatMessages");
  const id = "typing-" + Date.now();

  const wrapper = document.createElement("div");
  wrapper.className = "message bot";
  wrapper.id = id;

  const icon = document.createElement("div");
  icon.className = "bot-icon";
  icon.textContent = "🤖";

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<span></span><span></span><span></span>";

  wrapper.appendChild(icon);
  wrapper.appendChild(typing);
  container.appendChild(wrapper);
  scrollToBottom();

  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function setLoading(state) {
  isLoading = state;
  const btn = document.getElementById("sendBtn");
  btn.disabled = state;
}

function scrollToBottom() {
  const container = document.getElementById("chatMessages");
  container.scrollTop = container.scrollHeight;
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

function addToHistory(message) {
  const list = document.getElementById("historyList");
  const item = document.createElement("div");
  item.className = "history-item active";
  item.textContent = message.slice(0, 20) + (message.length > 20 ? "..." : "");

  // 이전 active 제거
  document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));

  list.prepend(item);
}

function newChat() {
  const container = document.getElementById("chatMessages");
  container.innerHTML = `
    <div class="welcome">
      <h2>한국어 챗봇</h2>
      <p>무엇이든 질문해보세요!</p>
    </div>
  `;
  chatHistory = [];
  document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));
}
