/* static/css/style.css is the primary stylesheet.
   This file holds minor JS-driven interactions. */

// ── Global CSRF Header for Fetch & XHR ──────────────────────────────────────────
(function setupCSRF() {
  const getCsrfToken = () => document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');

  // Intercept window.fetch
  if (window.fetch) {
    const originalFetch = window.fetch;
    window.fetch = function(url, options = {}) {
      const token = getCsrfToken();
      if (token) {
        options.headers = options.headers || {};
        if (options.headers instanceof Headers) {
          if (!options.headers.has('X-CSRFToken')) {
            options.headers.set('X-CSRFToken', token);
          }
        } else if (Array.isArray(options.headers)) {
          if (!options.headers.some(([k]) => k.toLowerCase() === 'x-csrftoken')) {
            options.headers.push(['X-CSRFToken', token]);
          }
        } else {
          if (!options.headers['X-CSRFToken'] && !options.headers['x-csrftoken']) {
            options.headers['X-CSRFToken'] = token;
          }
        }
      }
      return originalFetch.call(this, url, options);
    };
  }

  // Intercept XMLHttpRequest
  if (window.XMLHttpRequest) {
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function() {
      const token = getCsrfToken();
      if (token) {
        try {
          this.setRequestHeader('X-CSRFToken', token);
        } catch (_) {}
      }
      return originalSend.apply(this, arguments);
    };
  }
})();

// ── Upload drag-and-drop ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const zone = document.querySelector('.upload-zone');
  const fileInput = document.getElementById('audio_file');

  if (zone && fileInput) {
    zone.addEventListener('click', () => fileInput.click());

    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        updateZoneLabel(files[0].name);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) updateZoneLabel(fileInput.files[0].name);
    });

    function updateZoneLabel(name) {
      const hint = zone.querySelector('.upload-hint');
      if (hint) hint.textContent = `Selected: ${name}`;
    }
  }

  // ── Upload form — show upload indicator on submit ─────────────────────────
  const uploadForm = document.getElementById('upload-form');
  const submitBtn  = document.getElementById('upload-submit-btn');
  const spinnerEl  = document.getElementById('upload-spinner');
  const pctEl      = document.getElementById('loading-percentage');
  const statusEl   = document.getElementById('loading-status-text');

  const attendeeList = document.getElementById('attendee-list');
  const addAttendee = document.getElementById('add-attendee');
  if (attendeeList && addAttendee) {
    addAttendee.addEventListener('click', () => {
      const row = attendeeList.querySelector('.attendee-row');
      if (!row) return;
      const copy = row.cloneNode(true);
      copy.querySelectorAll('input').forEach((input) => { input.value = ''; });
      attendeeList.appendChild(copy);
    });
  }

  if (uploadForm && submitBtn) {
    uploadForm.addEventListener('submit', () => {
      submitBtn.hidden = true;
      const cards = uploadForm.querySelectorAll('.card');
      cards.forEach(c => { c.hidden = true; });
      
      if (spinnerEl) spinnerEl.hidden = false;
      if (pctEl) pctEl.textContent = '…';
      if (statusEl) statusEl.textContent = 'Uploading meeting to server...';
    });
  }

  // ── Stage-driven Progress Poller for meeting results ───────────────────────
  const STAGE_CONFIG = {
    'queued':            { pct: 15, text: 'Queued for processing...' },
    'recording':         { pct: 10, text: 'Recording...' },
    'uploading':         { pct: 20, text: 'Uploading the recording...' },
    'saved':             { pct: 25, text: 'Recording saved. Waiting to transcribe...' },
    'transcribing':      { pct: 40, text: 'Transcribing audio recording...' },
    'analyzing':         { pct: 60, text: 'Analyzing the transcript...' },
    'detecting_tasks':   { pct: 55, text: 'Labelling decisions and action items...' },
    'summarizing':       { pct: 75, text: 'Generating meeting summary (BART)...' },
    'saving':            { pct: 85, text: 'Saving results...' },
    'generating_charts': { pct: 95, text: 'Generating charts and knowledge graph...' },
    'completed':         { pct: 100, text: 'Completed. Loading results...' },
    'failed':            { pct: 100, text: 'Failed' }
  };

  const statusContainer = document.getElementById('processing-container');
  if (statusContainer) {
    const meetingId = statusContainer.getAttribute('data-meeting-id');
    const stageText = document.getElementById('processing-stage-text');
    const progressBar = document.getElementById('processing-progress-bar');

    if (meetingId) {
      const pollTimer = setInterval(() => {
        fetch(`/meeting/${encodeURIComponent(meetingId)}/status`)
          .then(res => {
            if (res.status === 404) {
              clearInterval(pollTimer);
              return null;
            }
            return res.json();
          })
          .then(data => {
            if (!data) return;

            if (data.status === 'completed') {
              clearInterval(pollTimer);
              if (progressBar) progressBar.style.width = '100%';
              if (stageText) stageText.textContent = 'Completed. Loading results...';
              window.location.reload();
            } else if (data.status === 'failed') {
              clearInterval(pollTimer);
              window.location.reload();
            } else {
              const stageKey = data.stage || data.status;
              const cfg = STAGE_CONFIG[stageKey] || { pct: 20, text: 'Processing...' };
              if (progressBar) progressBar.style.width = cfg.pct + '%';
              if (stageText) stageText.textContent = cfg.text;
            }
          })
          .catch(err => {
            console.warn('Status poll warning:', err);
          });
      }, 2000);
    }
  }

  // ── Auto-dismiss flash messages ─────────────────────────────────────────────
  document.querySelectorAll('.alert').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s ease';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 5000);
  });

  const navToggle = document.getElementById('nav-toggle');
  const primaryNav = document.getElementById('primary-nav');
  if (navToggle && primaryNav) {
    navToggle.addEventListener('click', () => {
      const open = primaryNav.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
  }

  const current = window.location.pathname;
  document.querySelectorAll('.navbar-nav a[href]').forEach(a => {
    if (a.getAttribute('href') === current) a.classList.add('active');
  });

  // ── Image Lightbox / Modal ────────────────────────────────────────────────────
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('image-modal-content');
  const closeBtn = document.getElementById('image-modal-close');
  const graphImages = document.querySelectorAll('.graph-img');

  if (modal && modalImg && closeBtn && graphImages.length > 0) {
    // Open modal when a graph is clicked
    graphImages.forEach(img => {
      img.addEventListener('click', function() {
        modal.style.display = 'block';
        modalImg.src = this.src;
      });
    });

    // Close modal when back button is clicked
    closeBtn.addEventListener('click', function() {
      modal.style.display = 'none';
    });

    // Close modal when clicking outside the image
    modal.addEventListener('click', function(e) {
      if (e.target === modal) {
        modal.style.display = 'none';
      }
    });

    // Close modal on escape key
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && modal.style.display === 'block') {
        modal.style.display = 'none';
      }
    });
  }

});
