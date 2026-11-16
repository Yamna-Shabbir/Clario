(function () {
  const form = document.getElementById("assistant-form");
  const thread = document.getElementById("assistant-thread");
  const error = document.getElementById("assistant-error");
  const input = document.getElementById("assistant-question");
  if (!form || !thread || !input) return;

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showError(text) {
    if (!error) return;
    error.hidden = !text;
    error.textContent = text || "";
  }

  function bubble(role, content) {
    const article = document.createElement("article");
    article.className = "assistant-bubble assistant-bubble--" + role;
    article.innerHTML =
      '<p class="assistant-bubble__label">' + (role === "user" ? "You" : "Clario") + "</p>" +
      '<div class="assistant-bubble__text"></div>';
    article.querySelector(".assistant-bubble__text").textContent = content;
    return article;
  }

  function evidenceList(items, messageId) {
    if (!items || !items.length) return null;
    const list = document.createElement("ul");
    list.className = "assistant-evidence";
    items.forEach(function (item, index) {
      const li = document.createElement("li");
      const bits = ['<span class="badge badge-lilac">' + escapeHtml(item.kind) + "</span>"];
      if (item.speaker) bits.push("<strong>" + escapeHtml(item.speaker) + "</strong>");
      if (item.timestamp) bits.push('<span class="text-muted">' + escapeHtml(item.timestamp) + "</span>");
      bits.push("<p>" + escapeHtml(item.excerpt) + "</p>");
      if (item.meeting_url) {
        bits.push('<a href="' + escapeHtml(item.meeting_url) + '">Open ' + escapeHtml(item.meeting_title || "meeting") + "</a>");
      }
      if (item.audio_url && item.seek !== null && item.seek !== undefined) {
        const audioId = "evidence-audio-live-" + messageId + "-" + index;
        bits.push('<audio id="' + audioId + '" preload="none" src="' + escapeHtml(item.audio_url) + '"></audio>');
        bits.push(
          '<button type="button" class="btn btn-outline btn-sm" data-evidence-play data-audio="' +
          audioId + '" data-start="' + escapeHtml(item.seek) + '">Play evidence</button>'
        );
      }
      li.innerHTML = bits.join(" ");
      list.appendChild(li);
    });
    return list;
  }

  async function ensureConversation() {
    const existing = form.getAttribute("data-conversation");
    if (existing) return existing;
    const response = await fetch(form.getAttribute("data-create-url"), {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) throw new Error("The conversation could not be started.");
    const payload = await response.json();
    form.setAttribute("data-conversation", String(payload.id));
    const next = new URL(window.location.href);
    next.searchParams.set("c", String(payload.id));
    window.history.replaceState({}, "", next);
    const list = document.querySelector(".assistant-history__list");
    const aside = document.querySelector(".assistant-history");
    if (aside && !list) {
      const empty = aside.querySelector(".text-muted");
      if (empty) empty.remove();
      const created = document.createElement("ul");
      created.className = "assistant-history__list";
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = next.pathname + next.search;
      link.className = "is-active";
      link.textContent = "New conversation";
      item.appendChild(link);
      created.appendChild(item);
      aside.appendChild(created);
    }
    return String(payload.id);
  }

  async function send(question) {
    showError("");
    const empty = document.getElementById("assistant-empty");
    if (empty) empty.remove();
    thread.appendChild(bubble("user", question));
    const conversationId = await ensureConversation();
    const response = await fetch("/assistant/conversations/" + conversationId + "/messages", {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const payload = await response.json().catch(function () { return {}; });
    if (!response.ok) {
      showError(payload.error || "The assistant could not answer.");
      return;
    }
    const article = bubble("assistant", payload.message.content || "");
    const list = evidenceList(payload.message.evidence, payload.message.id || "new");
    if (list) article.appendChild(list);
    if (payload.message.sources && payload.message.sources.length) {
      const note = document.createElement("p");
      note.className = "text-muted assistant-sources";
      note.textContent = payload.message.sources.map(function (source) { return source.source; }).join(", ");
      article.appendChild(note);
    }
    thread.appendChild(article);
    thread.scrollTop = thread.scrollHeight;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    send(question).catch(function () {
      showError("The assistant could not answer.");
    });
  });

  document.querySelectorAll("[data-suggestion]").forEach(function (button) {
    button.addEventListener("click", function () {
      input.value = button.getAttribute("data-suggestion") || "";
      form.requestSubmit();
    });
  });
})();
