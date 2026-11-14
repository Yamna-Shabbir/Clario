/**
 * Live meeting: WebRTC between participants, chat, audio-only transcript recording.
 */
(function () {
  const cfg = window.CLARIO_LIVE;
  if (!cfg || !cfg.urls) return;

  const stateEl = document.getElementById("record-state");
  const startBtn = document.getElementById("record-start");
  const stopBtn = document.getElementById("record-stop");
  const micBtn = document.getElementById("btn-mic");
  const camBtn = document.getElementById("btn-camera");
  const localVideo = document.getElementById("local-video");
  const videoPlaceholder = document.getElementById("video-placeholder");
  const selfTile = document.getElementById("self-tile");
  const rosterEl = document.getElementById("meet-roster");
  const remoteTiles = document.getElementById("remote-tiles");
  const chatLog = document.getElementById("chat-log");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const copyBtn = document.getElementById("copy-link");

  const myUserId = cfg.myUserId;
  const iceServers = cfg.iceServers || [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
    { urls: "stun:stun2.l.google.com:19302" },
  ];
  let meetingActive = Boolean(cfg.inMeeting || cfg.isHost);
  let inMeetingPollsStarted = false;
  let admissionDenied = cfg.admissionStatus === "denied";

  let localStream = null;
  let micOn = false;
  let camOn = false;
  let recorder = null;
  let recordChunks = [];

  let lastMessageId = 0;
  const seenMessageIds = new Set();
  let lastSignalId = 0;
  const peers = new Map();
  const peerNames = new Map();
  const remoteStreams = new Map();
  const iceQueues = new Map();
  const peerBuild = new Map();
  const negotiating = new Map();
  const offerSentAt = new Map();
  const speakingWatch = new Map();
  const peerReady = new Set();
  let speakTimer = null;
  let audioCtx = null;
  let mediaPrimed = false;
  let signalsBootstrapped = false;
  let signalPollBusy = false;

  const AUDIO_CONSTRAINTS = {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };

  const VIDEO_CONSTRAINTS = {
    width: { ideal: 1280, max: 1920 },
    height: { ideal: 720, max: 1080 },
    facingMode: "user",
  };

  function csrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.content : "";
  }

  function setToolState(btn, on, labelOn, labelOff) {
    if (!btn) return;
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    btn.classList.toggle("meet-tool--on", on);
    btn.classList.toggle("meet-tool--off", !on);
    const label = btn.querySelector(".meet-tool__label");
    if (label) label.textContent = on ? labelOn : labelOff;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function showChatError(msg) {
    let el = document.getElementById("chat-error");
    if (!el && chatForm) {
      el = document.createElement("p");
      el.id = "chat-error";
      el.className = "meet-chat-error";
      chatForm.parentElement.insertBefore(el, chatForm);
    }
    if (el) el.textContent = msg || "";
  }

  function liveTrack(kind) {
    if (!localStream) return null;
    return localStream.getTracks().find((t) => t.kind === kind && t.readyState === "live") || null;
  }

  async function refreshLocalStream() {
    const wantAudio = micOn;
    const wantVideo = camOn;
    if (!localStream) localStream = new MediaStream();

    localStream.getAudioTracks().forEach((track) => {
      if (!wantAudio) {
        track.stop();
        localStream.removeTrack(track);
      }
    });
    localStream.getVideoTracks().forEach((track) => {
      if (!wantVideo) {
        track.stop();
        localStream.removeTrack(track);
      }
    });

    const needAudio = wantAudio && !liveTrack("audio");
    const needVideo = wantVideo && !liveTrack("video");
    if (needAudio && needVideo) {
      try {
        const both = await navigator.mediaDevices.getUserMedia({
          audio: AUDIO_CONSTRAINTS,
          video: VIDEO_CONSTRAINTS,
        });
        both.getTracks().forEach((track) => localStream.addTrack(track));
      } catch (err) {
        /* try the mic and camera separately below */
      }
    }
    if (wantAudio && !liveTrack("audio")) {
      try {
        const audioStream = await navigator.mediaDevices.getUserMedia({
          audio: AUDIO_CONSTRAINTS,
          video: false,
        });
        audioStream.getAudioTracks().forEach((track) => localStream.addTrack(track));
      } catch (err) {
        if (stateEl && !(recorder && recorder.state === "recording")) {
          stateEl.textContent = "Microphone access was denied. Others can still see you if the camera is on.";
        }
      }
    }
    if (wantVideo && !liveTrack("video")) {
      try {
        const videoStream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: VIDEO_CONSTRAINTS,
        });
        videoStream.getVideoTracks().forEach((track) => localStream.addTrack(track));
      } catch (err) {
        if (stateEl && !(recorder && recorder.state === "recording")) {
          stateEl.textContent = liveTrack("audio")
            ? "Camera access was denied. Others can still hear you."
            : "Camera access was denied.";
        }
      }
    }

    if (localStream && !localStream.getTracks().length) localStream = null;
    micOn = Boolean(liveTrack("audio"));
    camOn = Boolean(liveTrack("video"));
    setToolState(micBtn, micOn, "Mic on", "Mic");
    setToolState(camBtn, camOn, "Camera on", "Camera");
    updateLocalPreview();
    const bindings = await bindAllPeers();
    for (const row of bindings) {
      if (row.needs) await requestRenegotiation(row.id);
    }
    syncPeersFromLastRoster();
    resumePlayback();
    if (mixRecorder && mixRecorder.state === "recording") refreshMixSources();
  }

  function updateLocalPreview() {
    if (!localVideo) return;
    localVideo.muted = true;
    const showVideo = Boolean(liveTrack("video"));
    if (showVideo) {
      localVideo.srcObject = localStream;
      localVideo.classList.add("meet-video--visible");
      if (videoPlaceholder) videoPlaceholder.hidden = true;
    } else {
      localVideo.classList.remove("meet-video--visible");
      if (videoPlaceholder) videoPlaceholder.hidden = false;
    }
  }

  async function toggleMic() {
    micOn = !micOn;
    setToolState(micBtn, micOn, "Mic on", "Mic");
    await refreshLocalStream();
  }

  async function toggleCamera() {
    camOn = !camOn;
    setToolState(camBtn, camOn, "Camera on", "Camera");
    await refreshLocalStream();
  }

  async function primeMeetingMedia() {
    if (mediaPrimed) return;
    mediaPrimed = true;
    micOn = true;
    camOn = true;
    setToolState(micBtn, true, "Mic on", "Mic");
    setToolState(camBtn, true, "Camera on", "Camera");
    await refreshLocalStream();
    if (stateEl && !(recorder && recorder.state === "recording")) {
      if (micOn && camOn) {
        stateEl.textContent = "Others can hear and see you. Recording separately saves audio for the transcript.";
      } else if (micOn) {
        stateEl.textContent = "Others can hear you. Turn the camera on if you want them to see you.";
      }
    }
  }

  function pickAudioMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
    ];
    for (const mime of candidates) {
      if (MediaRecorder.isTypeSupported(mime)) return mime;
    }
    return "";
  }

  function audioOnlyStream() {
    if (!localStream) return null;
    const tracks = localStream.getAudioTracks().filter((t) => t.readyState === "live");
    if (!tracks.length) return null;
    return new MediaStream(tracks);
  }

  let uploadTimer = null;
  let flushingForEnd = false;
  let mixDest = null;
  let mixSources = new Map();
  let mixRecorder = null;
  let mixChunks = [];

  function ensureMixDestination() {
    const ctx = audioContext();
    if (!ctx) return null;
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    if (!mixDest) mixDest = ctx.createMediaStreamDestination();
    return mixDest;
  }

  function connectTrackToMix(track) {
    if (!track || track.kind !== "audio" || track.readyState !== "live") return;
    if (mixSources.has(track.id)) return;
    const dest = ensureMixDestination();
    const ctx = audioContext();
    if (!dest || !ctx) return;
    try {
      const source = ctx.createMediaStreamSource(new MediaStream([track]));
      source.connect(dest);
      mixSources.set(track.id, source);
      track.addEventListener("ended", () => {
        try { source.disconnect(); } catch (err) { /* already closed */ }
        mixSources.delete(track.id);
      });
    } catch (err) {
      console.warn("Could not add a voice to the meeting recording", err);
    }
  }

  function refreshMixSources() {
    if (localStream) localStream.getAudioTracks().forEach(connectTrackToMix);
    remoteStreams.forEach((media) => {
      if (media && media.audio) media.audio.getAudioTracks().forEach(connectTrackToMix);
    });
  }

  function startMixRecorder() {
    refreshMixSources();
    const dest = ensureMixDestination();
    if (!dest || !dest.stream.getAudioTracks().length) return;
    if (mixRecorder && mixRecorder.state === "recording") return;
    const mime = pickAudioMimeType();
    mixChunks = [];
    try {
      mixRecorder = mime
        ? new MediaRecorder(dest.stream, { mimeType: mime, audioBitsPerSecond: 128000 })
        : new MediaRecorder(dest.stream);
    } catch (err) {
      try {
        mixRecorder = new MediaRecorder(dest.stream);
      } catch (again) {
        return;
      }
    }
    mixRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size) mixChunks.push(event.data);
    };
    mixRecorder.onstop = () => {
      uploadMix();
    };
    mixRecorder.start(1000);
  }

  async function uploadBlob(chunks, mimeType, role, keepChunks) {
    if (!chunks.length) return false;
    const type = mimeType || "audio/webm";
    const blob = new Blob(chunks, { type });
    if (!blob.size) return false;
    const extension = type.includes("ogg") ? "ogg" : "webm";
    const body = new FormData();
    body.append("audio", blob, (role === "mix" ? "meeting." : "speaker.") + extension);
    body.append("role", role);
    body.append("csrf_token", csrfToken());
    if (stateEl) {
      stateEl.textContent = role === "mix"
        ? "Uploading the whole meeting..."
        : "Uploading your microphone recording...";
    }
    const response = await fetch(cfg.urls.recording, {
      method: "POST",
      body,
      credentials: "same-origin",
    });
    let data = {};
    try {
      data = await response.json();
    } catch (e) {
      data = {};
    }
    if (!response.ok || !data.ok) {
      if (stateEl) stateEl.textContent = data.error || "The recording could not be saved.";
      return false;
    }
    if (!keepChunks && role === "speaker") recordChunks = [];
    if (!keepChunks && role === "mix") mixChunks = [];
    if (stateEl) {
      stateEl.textContent = role === "mix"
        ? "Saved. The meeting recording includes everyone who was connected."
        : "Saved. Your microphone is stored so the transcript can name you.";
    }
    pollRoster();
    return true;
  }

  function uploadRecording() {
    return uploadBlob(recordChunks, recorder && recorder.mimeType, "speaker");
  }

  function uploadMix() {
    return uploadBlob(mixChunks, mixRecorder && mixRecorder.mimeType, "mix");
  }

  async function beginRecording() {
    if (recorder && recorder.state === "recording") return;
    if (!micOn) await toggleMic();
    const stream = audioOnlyStream();
    if (!stream) {
      if (stateEl) stateEl.textContent = "Turn on your mic before recording.";
      return;
    }
    const mime = pickAudioMimeType();
    if (!mime) {
      if (stateEl) stateEl.textContent = "This browser does not support audio recording.";
      return;
    }
    recordChunks = [];
    try {
      recorder = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 128000 });
    } catch (err) {
      recorder = new MediaRecorder(stream);
    }
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) recordChunks.push(event.data);
    };
    recorder.onstop = () => {
      uploadRecording();
    };
    recorder.start(1000);
    startMixRecorder();
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
    if (stateEl) {
      stateEl.textContent = "Recording your microphone. Status: Recording.";
    }
    if (!uploadTimer) {
      uploadTimer = window.setInterval(() => {
        if (recorder && recorder.state === "recording" && recordChunks.length) {
          uploadBlob(recordChunks.slice(), recorder.mimeType, "speaker", true);
        }
        if (mixRecorder && mixRecorder.state === "recording" && mixChunks.length) {
          uploadBlob(mixChunks.slice(), mixRecorder.mimeType, "mix", true);
        }
      }, 12000);
    }
    if (selfTile) selfTile.classList.add("meet-tile--recording");
  }

  function stopRecording() {
    if (recorder && recorder.state === "recording") recorder.stop();
    if (mixRecorder && mixRecorder.state === "recording") mixRecorder.stop();
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    if (selfTile) selfTile.classList.remove("meet-tile--recording");
  }

  function renderRoster(members) {
    if (!rosterEl || !Array.isArray(members)) return;
    rosterEl.innerHTML = members
      .map(
        (m) => `<li data-user-id="${m.user_id}">
          <span class="meet-roster__name">${escapeHtml(m.display_name)}</span>
          <span class="text-muted meet-roster__email">${escapeHtml(m.email || "")}</span>
          <span class="badge ${m.recorded ? "badge-mint" : "badge-coral"} roster-rec">${m.recorded ? "Recorded" : "No recording"}</span>
        </li>`,
      )
      .join("");
  }

  let lastRoster = [];

  function syncPeersFromLastRoster() {
    if (lastRoster.length) syncPeers(lastRoster);
  }

  function renderWaitingList(waiting) {
    const list = document.getElementById("waiting-list");
    const empty = document.getElementById("waiting-empty");
    if (!list || !cfg.isHost) return;
    const rows = waiting || [];
    if (empty) empty.classList.toggle("hidden", rows.length > 0);
    list.innerHTML = rows
      .map(
        (m) => `<li data-user-id="${m.user_id}">
          <span>
            <span class="meet-roster__name">${escapeHtml(m.display_name)}</span>
            <span class="text-muted meet-roster__email">${escapeHtml(m.email || "")}</span>
          </span>
          <span class="meet-waiting-actions">
            <button type="button" class="btn btn-primary btn-sm" data-admit="${m.user_id}">Admit</button>
            <button type="button" class="btn btn-outline btn-sm" data-deny="${m.user_id}">Deny</button>
          </span>
        </li>`,
      )
      .join("");
  }

  const waitingListEl = document.getElementById("waiting-list");
  if (waitingListEl && cfg.isHost) {
    waitingListEl.addEventListener("click", (event) => {
      const admitBtn = event.target.closest("[data-admit]");
      const denyBtn = event.target.closest("[data-deny]");
      if (admitBtn) hostAdmit(parseInt(admitBtn.getAttribute("data-admit"), 10));
      if (denyBtn) hostDeny(parseInt(denyBtn.getAttribute("data-deny"), 10));
    });
  }

  async function hostAdmit(userId) {
    if (!cfg.urls.admit) return;
    const body = new FormData();
    body.append("user_id", String(userId));
    body.append("csrf_token", csrfToken());
    await fetch(cfg.urls.admit, {
      method: "POST",
      body,
      credentials: "same-origin",
      headers: { "X-Requested-With": "XMLHttpRequest" },
    });
    pollRoster();
  }

  async function hostDeny(userId) {
    if (!cfg.urls.deny) return;
    const body = new FormData();
    body.append("user_id", String(userId));
    body.append("csrf_token", csrfToken());
    await fetch(cfg.urls.deny, {
      method: "POST",
      body,
      credentials: "same-origin",
      headers: { "X-Requested-With": "XMLHttpRequest" },
    });
    pollRoster();
  }

  function setWaitingNotice(title, pulse) {
    const titleEl = document.getElementById("waiting-title");
    const pulseEl = document.getElementById("waiting-pulse");
    if (titleEl && title) titleEl.textContent = title;
    if (pulseEl && typeof pulse === "string") pulseEl.textContent = pulse;
  }

  async function parseJsonResponse(res) {
    const text = await res.text();
    try {
      return { data: JSON.parse(text), text };
    } catch (err) {
      return { data: null, text };
    }
  }

  async function registerViaRoster() {
    if (!cfg.urls.roster) return null;
    const res = await fetch(cfg.urls.roster, {
      method: "GET",
      credentials: "same-origin",
      cache: "no-store",
      headers: { Accept: "application/json", "X-Requested-With": "XMLHttpRequest" },
    });
    const { data, text } = await parseJsonResponse(res);
    if (data && data.ok) {
      return {
        ok: true,
        status: (data.my_admission || cfg.admissionStatus || "pending").toLowerCase(),
        registered: true,
      };
    }
    if (!res.ok && text.includes("CSRF")) {
      return { ok: false, error: "Session expired — refresh this page." };
    }
    return data || { ok: false, error: data?.error || `Server error (${res.status}).` };
  }

  async function registerJoinRequest() {
    const admissionUrl = cfg.urls.admission;
    if (admissionUrl) {
      const res = await fetch(admissionUrl, {
        method: "GET",
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json", "X-Requested-With": "XMLHttpRequest" },
      });
      const { data, text } = await parseJsonResponse(res);
      if (data && data.ok) return data;
      if (res.status === 404 || res.status === 405) {
        return registerViaRoster();
      }
      if (!res.ok && text.includes("CSRF")) {
        return { ok: false, error: "Session expired — refresh this page." };
      }
      if (!data?.ok) {
        const rosterResult = await registerViaRoster();
        if (rosterResult?.ok) return rosterResult;
      }
      return data || { ok: false, error: data?.error || `Server error (${res.status}).` };
    }
    return registerViaRoster();
  }

  async function sendJoinRequest() {
    if (meetingActive || cfg.isHost || admissionDenied) return false;
    const serverAlreadyWaiting =
      (cfg.admissionStatus || "").toLowerCase() === "pending" && !cfg.inMeeting;
    try {
      let data = await registerJoinRequest();
      if (!data?.ok && cfg.urls.request) {
        const body = new FormData();
        const token = csrfToken();
        if (token) body.append("csrf_token", token);
        const res = await fetch(cfg.urls.request, {
          method: "POST",
          body,
          credentials: "same-origin",
          cache: "no-store",
          headers: {
            Accept: "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRFToken": token,
          },
        });
        const parsed = await parseJsonResponse(res);
        data = parsed.data;
        if (!data?.ok && !res.ok) {
          if (res.status === 404 || res.status === 405) {
            data = await registerViaRoster();
          } else {
            data = { ok: false, error: data?.error || `Server error (${res.status}).` };
          }
        }
      }
      if (!data?.ok) {
        if (serverAlreadyWaiting) {
          setWaitingNotice("Request sent", "Waiting for the host to let you in…");
          return true;
        }
        setWaitingNotice(
          "Could not send request",
          data?.error || "Try refreshing this page.",
        );
        return false;
      }
      if ((data.status || "").toLowerCase() === "admitted") {
        handleAdmissionStatus("admitted");
        return true;
      }
      if ((data.status || "").toLowerCase() === "denied") {
        handleAdmissionStatus("denied");
        return false;
      }
      setWaitingNotice("Request sent", "Waiting for the host to let you in…");
      return true;
    } catch (err) {
      setWaitingNotice("Could not send request", "Check your connection and refresh.");
      return false;
    }
  }

  function showAdmissionDenied() {
    admissionDenied = true;
    const waitEl = document.getElementById("meet-waiting-screen");
    const mainEl = document.getElementById("meet-main");
    if (mainEl) mainEl.classList.add("hidden");
    if (!waitEl) return;
    waitEl.classList.remove("hidden");
    const title = document.getElementById("waiting-title");
    const lead = waitEl.querySelector(".meet-waiting__lead");
    const pulse = document.getElementById("waiting-pulse");
    if (title) title.textContent = "The host declined your request";
    if (lead) {
      lead.textContent =
        "You cannot join this meeting right now. You can close this page or ask the host to send the link again.";
    }
    if (pulse) pulse.textContent = "";
  }

  function startInMeetingPolls() {
    if (inMeetingPollsStarted) return;
    inMeetingPollsStarted = true;
    Promise.resolve(primeMeetingMedia())
      .catch((err) => console.warn("Meeting media setup failed", err));
    window.setInterval(pollMessages, 2500);
    window.setInterval(pollSignals, 800);
    window.setInterval(() => {
      peers.forEach((pc, id) => {
        const ice = pc.iceConnectionState;
        if (pc.connectionState === "failed" || ice === "failed" || ice === "disconnected") {
          maybeRenegotiate(id);
        }
      });
    }, 8000);
    pollMessages();
    pollSignals();
  }

  function enterMeeting() {
    if (meetingActive || cfg.isHost) return;
    meetingActive = true;
    const waitEl = document.getElementById("meet-waiting-screen");
    const mainEl = document.getElementById("meet-main");
    if (waitEl) waitEl.classList.add("hidden");
    if (mainEl) mainEl.classList.remove("hidden");
    startInMeetingPolls();
    pollRoster();
    maybeAutoRecord();
  }

  function handleAdmissionStatus(status) {
    if (meetingActive || cfg.isHost || admissionDenied) return;
    const normalized = (status || "").toLowerCase();
    if (normalized === "admitted") {
      enterMeeting();
    } else if (normalized === "denied") {
      showAdmissionDenied();
    }
  }

  async function pollAdmission() {
    if (meetingActive || cfg.isHost || admissionDenied || !cfg.urls.admission) return;
    try {
      const res = await fetch(cfg.urls.admission, {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await res.json();
      if (!data.ok) return;
      handleAdmissionStatus(data.status);
    } catch (err) {
      /* ignore */
    }
  }

  async function pollRoster() {
    try {
      const res = await fetch(cfg.urls.roster, {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await res.json();
      if (!data.ok) return;
      if (!meetingActive && !cfg.isHost && data.my_admission) {
        handleAdmissionStatus(data.my_admission);
      }
      if (Array.isArray(data.waiting)) renderWaitingList(data.waiting);
      if (data.status === "ending" && !flushingForEnd) {
        flushingForEnd = true;
        if (stateEl) stateEl.textContent = "The host is ending the meeting. Uploading your microphone.";
        stopRecording();
      }
      if (data.members && (meetingActive || cfg.isHost || data.status === "ending")) {
        lastRoster = data.members;
        renderRoster(data.members);
        if (meetingActive || cfg.isHost) syncPeers(data.members);
      }
      if (cfg.isHost && Array.isArray(data.waiting) && data.waiting.length > 0) {
        const wrap = document.getElementById("host-waiting-wrap");
        if (wrap) wrap.classList.add("meet-waiting-host--alert");
      }
    } catch (err) {
      /* ignore */
    }
  }

  function appendChatMessage(row) {
    if (!chatLog || !row) return;
    if (row.id && seenMessageIds.has(row.id)) return;
    if (row.id) seenMessageIds.add(row.id);
    const el = document.createElement("div");
    el.className = "meet-chat-line";
    el.innerHTML = `<strong>${escapeHtml(row.display_name)}</strong>
      <span class="text-muted">${escapeHtml(row.created_at || "")}</span>
      <p>${escapeHtml(row.message)}</p>`;
    chatLog.appendChild(el);
    chatLog.scrollTop = chatLog.scrollHeight;
    if (row.id) lastMessageId = Math.max(lastMessageId, row.id);
  }

  async function pollMessages() {
    try {
      const url = `${cfg.urls.messages}?since=${lastMessageId}`;
      const res = await fetch(url, { credentials: "same-origin" });
      const data = await res.json();
      if (data.ok && data.messages) data.messages.forEach(appendChatMessage);
    } catch (err) {
      /* ignore */
    }
  }

  async function sendChat(message) {
    const body = new FormData();
    body.append("message", message);
    body.append("csrf_token", csrfToken());
    const res = await fetch(cfg.urls.chat, {
      method: "POST",
      body,
      credentials: "same-origin",
    });
    let data = {};
    try {
      data = await res.json();
    } catch (e) {
      showChatError("Chat could not reach the server. Restart the app and try again.");
      return data;
    }
    if (!res.ok || !data.ok) {
      showChatError(data.error || "Message was not sent.");
      return data;
    }
    showChatError("");
    await pollMessages();
    return data;
  }

  function isPolite(remoteId) {
    return Number(myUserId) > Number(remoteId);
  }

  function updateGridCount() {
    const grid = document.getElementById("meet-stage-grid");
    if (!grid) return;
    const count = grid.querySelectorAll(".meet-tile").length;
    grid.setAttribute("data-count", String(Math.max(count, 1)));
  }

  function tileInitial(name) {
    const cleaned = (name || "?").trim();
    return (cleaned.charAt(0) || "?").toUpperCase();
  }

  function ensureRemoteTile(remoteId, remoteName) {
    const name = remoteName || peerNames.get(Number(remoteId)) || "Participant";
    let wrap = document.getElementById(`remote-tile-${remoteId}`);
    if (!wrap) {
      if (!remoteTiles) return null;
      wrap = document.createElement("div");
      wrap.id = `remote-tile-${remoteId}`;
      wrap.className = "meet-tile meet-tile--remote";
      const video = document.createElement("video");
      video.className = "meet-video meet-remote-video";
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "");
      const audio = document.createElement("audio");
      audio.className = "meet-remote-audio";
      audio.autoplay = true;
      const placeholder = document.createElement("div");
      placeholder.className = "meet-tile__placeholder";
      const avatar = document.createElement("span");
      avatar.className = "meet-avatar";
      const status = document.createElement("p");
      status.className = "meet-tile__status";
      status.textContent = "Connecting…";
      placeholder.appendChild(avatar);
      placeholder.appendChild(status);
      const label = document.createElement("span");
      label.className = "meet-tile__label";
      wrap.appendChild(video);
      wrap.appendChild(audio);
      wrap.appendChild(placeholder);
      wrap.appendChild(label);
      remoteTiles.appendChild(wrap);
    }
    const avatar = wrap.querySelector(".meet-avatar");
    const label = wrap.querySelector(".meet-tile__label");
    if (avatar) avatar.textContent = tileInitial(name);
    if (label) label.textContent = name;
    updateGridCount();
    return wrap;
  }

  function refreshTileMedia(remoteId) {
    const wrap = document.getElementById(`remote-tile-${remoteId}`);
    if (!wrap) return;
    const media = remoteStreams.get(Number(remoteId));
    const videoEl = wrap.querySelector("video");
    const placeholder = wrap.querySelector(".meet-tile__placeholder");
    const status = wrap.querySelector(".meet-tile__status");
    const videoTracks = media && media.video
      ? media.video.getVideoTracks().filter((track) => track.readyState === "live" && track.enabled)
      : [];
    const audioTracks = media && media.audio
      ? media.audio.getAudioTracks().filter((track) => track.readyState === "live" && track.enabled)
      : [];
    const frames = Boolean(videoEl && videoEl.videoWidth > 0);
    const showVideo = frames || videoTracks.some((track) => !track.muted);
    if (videoEl) {
      videoEl.muted = true;
      videoEl.classList.toggle("meet-video--visible", showVideo);
    }
    if (placeholder) placeholder.hidden = showVideo;
    if (status && !showVideo) {
      const pc = peers.get(Number(remoteId));
      const state = pc ? pc.connectionState : "new";
      if (state === "failed") status.textContent = "Reconnecting…";
      else if (state !== "connected") status.textContent = "Connecting…";
      else if (videoTracks.length) status.textContent = "Video starting…";
      else if (audioTracks.length) status.textContent = "Camera off";
      else status.textContent = "Waiting for their mic and camera…";
    }
  }

  function showHearBanner() {
    const button = document.getElementById("meet-hear");
    if (button) button.hidden = false;
  }

  function hideHearBanner() {
    const button = document.getElementById("meet-hear");
    if (!button) return;
    const waiting = Array.from(document.querySelectorAll(".meet-remote-audio")).some(
      (audio) => audio.srcObject && audio.paused,
    );
    if (!waiting) button.hidden = true;
  }

  function audioContext() {
    if (!audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      audioCtx = new Ctx();
    }
    return audioCtx;
  }

  function resumePlayback() {
    const ctx = audioContext();
    if (ctx && ctx.state === "suspended") ctx.resume().catch(() => {});
    document.querySelectorAll(".meet-remote-video").forEach((video) => {
      video.muted = true;
      if (video.srcObject) video.play().catch(() => {});
    });
    document.querySelectorAll(".meet-remote-audio").forEach((audio) => {
      if (audio.srcObject) playRemoteAudio(audio);
    });
  }

  function playRemoteAudio(audio) {
    audio.muted = false;
    audio.volume = 1;
    return audio.play().then(() => {
      hideHearBanner();
    }).catch(() => {
      showHearBanner();
    });
  }

  function watchSpeaking(remoteId, stream) {
    const id = Number(remoteId);
    if (speakingWatch.has(id) || !stream || !stream.getAudioTracks().length) return;
    const ctx = audioContext();
    if (!ctx) return;
    try {
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      speakingWatch.set(id, { analyser, source });
      if (!speakTimer) {
        const data = new Uint8Array(analyser.frequencyBinCount);
        speakTimer = window.setInterval(() => {
          speakingWatch.forEach((entry, speakerId) => {
            const wrap = document.getElementById(`remote-tile-${speakerId}`);
            if (!wrap) return;
            entry.analyser.getByteFrequencyData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i += 1) sum += data[i];
            wrap.classList.toggle("meet-tile--speaking", sum / data.length > 12);
          });
        }, 200);
      }
    } catch (err) {
      /* analyser is optional */
    }
  }

  function mediaFor(remoteId) {
    const id = Number(remoteId);
    let media = remoteStreams.get(id);
    if (!media || !media.audio || !media.video) {
      media = { audio: new MediaStream(), video: new MediaStream() };
      remoteStreams.set(id, media);
    }
    return media;
  }

  function attachRemoteTrack(remoteId, track) {
    const id = Number(remoteId);
    const media = mediaFor(id);
    const bucket = track.kind === "audio" ? media.audio : media.video;
    if (!bucket.getTracks().some((existing) => existing.id === track.id)) {
      bucket.addTrack(track);
    }
    const wrap = ensureRemoteTile(id, peerNames.get(id));
    const video = wrap ? wrap.querySelector("video") : null;
    const audio = wrap ? wrap.querySelector("audio") : null;
    if (track.kind === "video" && video && video.srcObject !== media.video) {
      video.muted = true;
      video.srcObject = media.video;
      video.onresize = () => refreshTileMedia(id);
      video.onloadeddata = () => refreshTileMedia(id);
      video.play().catch(() => {});
    }
    if (track.kind === "audio" && audio && audio.srcObject !== media.audio) {
      audio.srcObject = media.audio;
      playRemoteAudio(audio);
    }
    const onChange = () => refreshTileMedia(id);
    track.onunmute = onChange;
    track.onmute = onChange;
    track.onended = () => {
      if (bucket.getTracks().some((existing) => existing.id === track.id)) {
        bucket.removeTrack(track);
      }
      refreshTileMedia(id);
    };
    refreshTileMedia(id);
    if (track.kind === "audio") {
      watchSpeaking(id, media.audio);
      if (mixRecorder && mixRecorder.state === "recording") connectTrackToMix(track);
    }
  }

  async function sendSignal(toUserId, kind, payload) {
    if (!cfg.urls.signal) return;
    const body = new FormData();
    body.append("to_user_id", String(toUserId));
    body.append("kind", kind);
    body.append("payload", JSON.stringify(payload));
    body.append("csrf_token", csrfToken());
    const response = await fetch(cfg.urls.signal, {
      method: "POST",
      body,
      credentials: "same-origin",
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRFToken": csrfToken(),
      },
    });
    let data = {};
    try {
      data = await response.json();
    } catch (err) {
      data = {};
    }
    if (!response.ok || !data.ok) {
      throw new Error((data && data.error) || `Signal failed (${response.status})`);
    }
  }

  function findTransceiver(pc, kind) {
    return pc.getTransceivers().find((transceiver) => {
      const senderTrack = transceiver.sender && transceiver.sender.track;
      if (senderTrack) return senderTrack.kind === kind;
      const receiverTrack = transceiver.receiver && transceiver.receiver.track;
      return Boolean(receiverTrack && receiverTrack.kind === kind);
    }) || null;
  }

  async function bindLocalToPeer(pc) {
    let needsOffer = false;
    for (const kind of ["audio", "video"]) {
      const track = liveTrack(kind);
      let transceiver = findTransceiver(pc, kind);
      if (!transceiver && track && localStream) {
        try {
          pc.addTrack(track, localStream);
          needsOffer = true;
          continue;
        } catch (err) {
          console.warn("addTrack", kind, err);
        }
      }
      if (!transceiver) continue;
      if (!track) {
        if (transceiver.sender && transceiver.sender.track) {
          try {
            await transceiver.sender.replaceTrack(null);
          } catch (err) {
            console.warn("clearTrack", kind, err);
          }
        }
        continue;
      }
      const direction = transceiver.currentDirection || transceiver.direction;
      const canSend = direction === "sendrecv" || direction === "sendonly";
      if (transceiver.direction !== "sendrecv") {
        try {
          transceiver.direction = "sendrecv";
        } catch (err) {
          /* older browsers */
        }
      }
      if (transceiver.sender.track !== track) {
        try {
          await transceiver.sender.replaceTrack(track);
        } catch (err) {
          console.warn("replaceTrack", kind, err);
        }
      }
      if (!canSend) needsOffer = true;
    }
    return needsOffer;
  }

  async function bindAllPeers() {
    const jobs = [];
    peers.forEach((pc, id) => {
      jobs.push(bindLocalToPeer(pc).then((needs) => ({ id, needs })));
    });
    return Promise.all(jobs);
  }

  async function requestRenegotiation(remoteId) {
    const id = Number(remoteId);
    const pc = peers.get(id);
    if (!pc || !peerReady.has(id)) return;
    if (!isPolite(id)) {
      await negotiate(pc, id);
      return;
    }
    try {
      await sendSignal(id, "refresh", { at: Date.now() });
    } catch (err) {
      console.warn("Renegotiate request failed", err);
    }
  }

  async function negotiate(pc, remoteId) {
    const id = Number(remoteId);
    if (isPolite(id) || negotiating.get(id)) return;
    negotiating.set(id, true);
    offerSentAt.set(id, Date.now());
    try {
      if (pc.signalingState === "have-local-offer") {
        await pc.setLocalDescription({ type: "rollback" });
      }
      if (pc.signalingState !== "stable") return;
      const restart = pc.connectionState === "failed" || pc.iceConnectionState === "failed";
      const offer = await pc.createOffer(restart ? { iceRestart: true } : undefined);
      if (pc.signalingState !== "stable") return;
      await pc.setLocalDescription(offer);
      await sendSignal(id, "offer", {
        type: pc.localDescription.type,
        sdp: pc.localDescription.sdp,
      });
    } catch (err) {
      console.warn("WebRTC offer failed", err);
    } finally {
      negotiating.set(id, false);
    }
  }

  function maybeRenegotiate(remoteId) {
    const id = Number(remoteId);
    const pc = peers.get(id);
    if (!pc || !peerReady.has(id) || isPolite(id)) return;
    if (pc.connectionState === "connected") return;
    if (Date.now() - (offerSentAt.get(id) || 0) < 3000) return;
    const waitingForAnswer = !(pc.remoteDescription && pc.remoteDescription.type);
    const failed = pc.connectionState === "failed" || pc.iceConnectionState === "failed";
    if (waitingForAnswer || failed) negotiate(pc, id);
  }

  async function addCandidate(remoteId, pc, candidate) {
    if (!candidate) return;
    const id = Number(remoteId);
    if (!pc.remoteDescription || !pc.remoteDescription.type) {
      const queued = iceQueues.get(id) || [];
      queued.push(candidate);
      iceQueues.set(id, queued);
      return;
    }
    try {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
      console.warn("ICE candidate failed", err);
    }
  }

  async function flushCandidates(remoteId, pc) {
    const id = Number(remoteId);
    const queued = iceQueues.get(id) || [];
    iceQueues.set(id, []);
    for (const candidate of queued) {
      try {
        await pc.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (err) {
        console.warn("Queued ICE candidate failed", err);
      }
    }
  }

  async function buildPeer(remoteId, remoteName) {
    const id = Number(remoteId);
    if (peers.has(id)) return peers.get(id);
    peerNames.set(id, remoteName);
    ensureRemoteTile(id, remoteName);
    const pc = new RTCPeerConnection({
      iceServers,
      iceCandidatePoolSize: 4,
    });
    peers.set(id, pc);
    pc.onicecandidate = (event) => {
      if (event.candidate) sendSignal(id, "ice", event.candidate.toJSON()).catch(() => {});
    };
    pc.ontrack = (event) => {
      if (event.track) attachRemoteTrack(id, event.track);
    };
    pc.onconnectionstatechange = () => {
      refreshTileMedia(id);
      if (pc.connectionState === "failed") maybeRenegotiate(id);
    };
    if (!isPolite(id)) {
      // Offerer owns the m-lines. Polite peer answers the remote offer instead.
      pc.addTransceiver("audio", { direction: "sendrecv" });
      pc.addTransceiver("video", { direction: "sendrecv" });
      await bindLocalToPeer(pc);
      peerReady.add(id);
      await negotiate(pc, id);
    } else {
      peerReady.add(id);
    }
    return pc;
  }

  function ensurePeer(remoteId, remoteName) {
    const id = Number(remoteId);
    peerNames.set(id, remoteName);
    ensureRemoteTile(id, remoteName);
    if (peers.has(id)) {
      maybeRenegotiate(id);
      return peers.get(id);
    }
    if (!peerBuild.has(id)) {
      const job = buildPeer(id, remoteName).catch((err) => {
        console.warn("Peer setup failed", err);
        peerBuild.delete(id);
        peers.delete(id);
      });
      peerBuild.set(id, job);
    }
    return peerBuild.get(id);
  }

  function teardownPeer(remoteId) {
    const id = Number(remoteId);
    const pc = peers.get(id);
    if (pc) pc.close();
    peers.delete(id);
    peerBuild.delete(id);
    peerReady.delete(id);
    iceQueues.delete(id);
    offerSentAt.delete(id);
    negotiating.delete(id);
    const speak = speakingWatch.get(id);
    if (speak) {
      try { speak.source.disconnect(); } catch (err) { /* already closed */ }
      speakingWatch.delete(id);
    }
    remoteStreams.delete(id);
    peerNames.delete(id);
    const tile = document.getElementById(`remote-tile-${id}`);
    if (tile) tile.remove();
    updateGridCount();
  }

  function signalPayload(payload) {
    if (typeof payload === "string") {
      try {
        return JSON.parse(payload);
      } catch (err) {
        return payload;
      }
    }
    return payload;
  }

  async function handleSignal(row) {
    const fromId = Number(row.from_user_id);
    const kind = row.kind;
    const payload = signalPayload(row.payload);
    if (!peers.has(fromId)) {
      await ensurePeer(fromId, peerNames.get(fromId) || "Participant");
    }
    const pc = peers.get(fromId);
    if (!pc) return;
    if (kind === "offer") {
      if (pc.signalingState !== "stable") {
        if (!isPolite(fromId)) return;
        try {
          await pc.setLocalDescription({ type: "rollback" });
        } catch (err) {
          console.warn("Rollback before answer failed", err);
        }
      }
      await pc.setRemoteDescription(new RTCSessionDescription(payload));
      await flushCandidates(fromId, pc);
      await bindLocalToPeer(pc);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await sendSignal(fromId, "answer", {
        type: pc.localDescription.type,
        sdp: pc.localDescription.sdp,
      });
      refreshTileMedia(fromId);
      resumePlayback();
    } else if (kind === "refresh") {
      if (!isPolite(fromId)) await negotiate(pc, fromId);
    } else if (kind === "answer") {
      if (pc.signalingState !== "have-local-offer") return;
      await pc.setRemoteDescription(new RTCSessionDescription(payload));
      await flushCandidates(fromId, pc);
      await bindLocalToPeer(pc);
      refreshTileMedia(fromId);
      resumePlayback();
    } else if (kind === "ice") {
      await addCandidate(fromId, pc, payload);
    }
  }

  async function pollSignals() {
    if (!cfg.urls.signal || signalPollBusy) return;
    signalPollBusy = true;
    try {
      if (!signalsBootstrapped) {
        const boot = await fetch(`${cfg.urls.signal}?bootstrap=1`, {
          credentials: "same-origin",
          cache: "no-store",
          headers: { Accept: "application/json" },
        });
        const bootData = await boot.json();
        if (bootData.ok && Number.isFinite(Number(bootData.cursor))) {
          lastSignalId = Number(bootData.cursor);
          signalsBootstrapped = true;
        }
        return;
      }
      const res = await fetch(`${cfg.urls.signal}?since=${lastSignalId}`, {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await res.json();
      if (!data.ok || !data.signals) return;
      for (const row of data.signals) {
        try {
          await handleSignal(row);
          lastSignalId = Math.max(lastSignalId, row.id);
        } catch (err) {
          console.warn("Signal handling failed", row.kind, err);
          if (row.kind === "ice") {
            lastSignalId = Math.max(lastSignalId, row.id);
            continue;
          }
          break;
        }
      }
    } catch (err) {
      /* ignore */
    } finally {
      signalPollBusy = false;
    }
  }

  function syncPeers(members) {
    if (!meetingActive && !cfg.isHost) return;
    const remoteIds = new Set();
    members.forEach((member) => {
      const id = Number(member.user_id);
      if (id === Number(myUserId)) return;
      remoteIds.add(id);
      peerNames.set(id, member.display_name);
      ensureRemoteTile(id, member.display_name);
      ensurePeer(id, member.display_name);
    });
    Array.from(peers.keys()).forEach((id) => {
      if (!remoteIds.has(id)) teardownPeer(id);
    });
  }

  document.querySelectorAll(".meet-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const panel = tab.getAttribute("data-panel");
      document.querySelectorAll(".meet-tab").forEach((t) => {
        const active = t === tab;
        t.classList.toggle("meet-tab--active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".meet-panel").forEach((p) => {
        const isPeople = p.id === "panel-people";
        const show = (panel === "people" && isPeople) || (panel === "chat" && p.id === "panel-chat");
        p.classList.toggle("meet-panel--active", show);
        p.hidden = !show;
      });
      if (panel === "chat") pollMessages();
    });
  });

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const link = document.getElementById("join-link");
      try {
        await navigator.clipboard.writeText(link.value);
        copyBtn.textContent = "Copied";
      } catch (err) {
        link.select();
        copyBtn.textContent = "Select";
      }
    });
  }

  if (micBtn) micBtn.addEventListener("click", toggleMic);
  if (camBtn) camBtn.addEventListener("click", toggleCamera);
  const hearBtn = document.getElementById("meet-hear");
  if (hearBtn) hearBtn.addEventListener("click", resumePlayback);
  document.addEventListener("pointerdown", () => {
    if (meetingActive || cfg.isHost) resumePlayback();
  });
  updateGridCount();
  const consentBox = document.getElementById("record-consent");
  function maybeAutoRecord() {
    if (!consentBox || !consentBox.checked) return;
    if (!meetingActive && !cfg.isHost) return;
    beginRecording();
  }
  if (consentBox) {
    consentBox.addEventListener("change", () => {
      if (consentBox.checked) maybeAutoRecord();
      else stopRecording();
    });
  }
  if (startBtn) {
    startBtn.addEventListener("click", () => {
      if (consentBox && !consentBox.checked) {
        if (stateEl) stateEl.textContent = "Agree to record your microphone before recording starts.";
        return;
      }
      beginRecording();
    });
  }
  if (stopBtn) stopBtn.addEventListener("click", stopRecording);

  const endForm = document.querySelector(".meet-end-form");
  if (endForm) {
    endForm.addEventListener("submit", (event) => {
      const recordingNow = recorder && recorder.state === "recording";
      const mixingNow = mixRecorder && mixRecorder.state === "recording";
      event.preventDefault();
      const pending = [];
      if (recordingNow) {
        pending.push(new Promise((resolve) => {
          recorder.onstop = async () => resolve(await uploadRecording());
          recorder.stop();
        }));
      }
      if (mixingNow) {
        pending.push(new Promise((resolve) => {
          mixRecorder.onstop = async () => resolve(await uploadMix());
          mixRecorder.stop();
        }));
      }
      Promise.all(pending).then(async (saved) => {
        if ((recordingNow || mixingNow) && !saved.some((ok) => ok)) {
          if (stateEl) stateEl.textContent = "Your microphone was not saved, so the meeting was not ended.";
          return;
        }
        if (cfg.urls.ending) {
          const body = new FormData();
          body.append("csrf_token", csrfToken());
          await fetch(cfg.urls.ending, { method: "POST", body, credentials: "same-origin" });
          const deadline = Date.now() + 15000;
          while (Date.now() < deadline) {
            const res = await fetch(cfg.urls.roster, { credentials: "same-origin", cache: "no-store" });
            const data = await res.json();
            const members = data.members || [];
            if (!members.length || members.every((member) => member.recorded)) break;
            if (stateEl) {
              const waiting = members.filter((member) => !member.recorded).map((member) => member.display_name);
              stateEl.textContent = "Waiting for microphones: " + waiting.join(", ");
            }
            await new Promise((resolve) => window.setTimeout(resolve, 1000));
          }
        }
        endForm.submit();
      });
    });
  }
  window.addEventListener("pagehide", () => {
    if (recorder && recorder.state === "recording") {
      recorder.onstop = () => {};
      try { recorder.requestData(); } catch (err) { /* flush is best-effort */ }
      try { recorder.stop(); } catch (err) { /* the page is going away */ }
    }
    if (mixRecorder && mixRecorder.state === "recording") {
      mixRecorder.onstop = () => {};
      try { mixRecorder.requestData(); } catch (err) { /* flush is best-effort */ }
      try { mixRecorder.stop(); } catch (err) { /* the page is going away */ }
    }
    const beaconAudio = (chunks, mimeType, role) => {
      if (!chunks.length || !cfg.urls.recording || !navigator.sendBeacon) return;
      const type = mimeType || "audio/webm";
      const blob = new Blob(chunks, { type });
      if (!blob.size) return;
      const audioBody = new FormData();
      audioBody.append("csrf_token", csrfToken());
      audioBody.append("role", role);
      audioBody.append("audio", blob, (role === "mix" ? "meeting." : "speaker.") + (type.includes("ogg") ? "ogg" : "webm"));
      navigator.sendBeacon(cfg.urls.recording, audioBody);
    };
    beaconAudio(recordChunks, recorder && recorder.mimeType, "speaker");
    beaconAudio(mixChunks, mixRecorder && mixRecorder.mimeType, "mix");
    if (!cfg.urls.leave || !navigator.sendBeacon) return;
    const body = new FormData();
    body.append("csrf_token", csrfToken());
    navigator.sendBeacon(cfg.urls.leave, body);
  });
  if (meetingActive || cfg.isHost) maybeAutoRecord();

  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = (chatInput.value || "").trim();
      if (!text) return;
      chatInput.value = "";
      await sendChat(text);
    });
  }

  const rosterMs = !meetingActive && !cfg.isHost ? 1500 : 2000;
  window.setInterval(pollRoster, rosterMs);
  pollRoster();

  if (!meetingActive && !cfg.isHost && !admissionDenied) {
    sendJoinRequest().then(() => {
      window.setInterval(pollAdmission, 1500);
      pollAdmission();
    });
  } else if (meetingActive || cfg.isHost) {
    startInMeetingPolls();
  }
})();
