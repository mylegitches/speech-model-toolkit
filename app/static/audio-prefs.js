/*
 * Audio preferences shared by every tab (loaded by the tool pages).
 *
 * Microphone, speaker and volume are per browser, so they live in
 * localStorage (all tabs share one origin). Everything else in Settings →
 * Audio is stored on the server and fetched with SMT.serverSettings().
 */
(function () {
  const KEYS = {
    input: 'smt.audio.inputDeviceId',
    output: 'smt.audio.outputDeviceId',
    volume: 'smt.audio.volume',
  };

  function get(name, fallback = '') {
    try {
      const value = localStorage.getItem(KEYS[name]);
      return value === null ? fallback : value;
    } catch (e) {
      return fallback;
    }
  }

  function set(name, value) {
    try {
      localStorage.setItem(KEYS[name], value);
    } catch (e) { /* private mode: not remembered */ }
  }

  // Tool pages live one level down (/wakeword/, /voice/, /lab/, /settings/)
  async function serverSettings() {
    try {
      const res = await fetch('../settings/api/settings');
      return res.ok ? await res.json() : null;
    } catch (e) {
      return null;
    }
  }

  /** getUserMedia with the preferred microphone, falling back to the default one. */
  async function openMic(constraints = {}, deviceId = get('input')) {
    const audio = { channelCount: 1, ...constraints };
    if (deviceId) {
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: { ...audio, deviceId: { exact: deviceId } },
        });
      } catch (e) {
        if (e.name !== 'OverconstrainedError' && e.name !== 'NotFoundError') throw e;
      }
    }
    return navigator.mediaDevices.getUserMedia({ audio });
  }

  /** Route an <audio> element to the preferred speaker and volume. */
  async function applyOutput(audioEl) {
    if (!audioEl) return;
    audioEl.volume = Math.max(0, Math.min(1, Number(get('volume', '1')) || 0));
    const sink = get('output');
    if (sink && typeof audioEl.setSinkId === 'function') {
      try {
        await audioEl.setSinkId(sink);
      } catch (e) {
        console.warn('Could not use the chosen speaker:', e);
      }
    }
  }

  /** Allow or block the browser's own "Download" on an <audio controls> element. */
  function applyDownloads(audioEl, allowed) {
    if (!audioEl) return;
    if (allowed) {
      audioEl.removeAttribute('controlsList');
      audioEl.oncontextmenu = null;
    } else {
      audioEl.setAttribute('controlsList', 'nodownload');
      audioEl.oncontextmenu = (e) => e.preventDefault();
    }
  }

  // ---- Mobile helpers ------------------------------------------------------

  const SILENCE = 'data:audio/wav;base64,UklGRjQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YRAAAAAAAAAAAAAAAAAAAAAAAAAA';

  /**
   * Phones (iOS Safari especially) only let an <audio> element play sound that
   * starts from a tap. Call this synchronously inside the click handler; the
   * element can then play later, e.g. when a reply arrives.
   */
  function unlockAudio(audioEl) {
    if (!audioEl || audioEl.dataset.unlocked) return;
    audioEl.dataset.unlocked = '1';
    const src = audioEl.getAttribute('src');
    audioEl.src = SILENCE;
    const done = () => {
      if (audioEl.src !== SILENCE) return;  // real audio was set meanwhile
      if (src) audioEl.src = src;
      else audioEl.removeAttribute('src');
    };
    audioEl.play().then(() => { audioEl.pause(); done(); }).catch(done);
  }

  /** Keep the screen on while listening (the phone would otherwise sleep and cut the mic). */
  let wakeLock = null;
  let wantAwake = false;
  async function keepAwake(on) {
    wantAwake = on;
    if (!('wakeLock' in navigator)) return;
    try {
      if (on && !wakeLock) {
        wakeLock = await navigator.wakeLock.request('screen');
        wakeLock.addEventListener('release', () => { wakeLock = null; });
      } else if (!on && wakeLock) {
        await wakeLock.release();
        wakeLock = null;
      }
    } catch (e) { /* not allowed right now (e.g. page hidden) */ }
  }
  // The lock is dropped when the page is hidden; take it again on return
  document.addEventListener('visibilitychange', () => {
    if (wantAwake && document.visibilityState === 'visible') keepAwake(true);
  });

  // ---- Errors ----------------------------------------------------------------

  /** A readable message for a failed response (server text, JSON detail, or proxy trouble). */
  async function errorMessage(res) {
    if (res.status === 413) {
      return 'The file is too large for the server or your reverse proxy (raise its upload limit, e.g. nginx client_max_body_size).';
    }
    if ([502, 503, 504].includes(res.status)) {
      return `The server didn't respond (HTTP ${res.status}). It may be restarting, or a reverse proxy timed out. Try again in a moment.`;
    }
    let text = '';
    try { text = await res.text(); } catch (e) { /* no body */ }
    try {
      const data = JSON.parse(text);
      const detail = data.detail ?? data.error ?? data.message;
      if (Array.isArray(detail)) text = detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
      else if (detail) text = String(detail);
    } catch (e) { /* plain text */ }
    return text.trim() || `Request failed (HTTP ${res.status} ${res.statusText})`;
  }

  /** fetch that throws Error(readable message) and returns JSON (or a Blob for audio/files). */
  async function request(url, options = {}) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw new Error("Can't reach the server. Check your connection and that the container is running.");
    }
    if (!res.ok) throw new Error(await errorMessage(res));
    const type = res.headers.get('content-type') || '';
    return type.includes('json') ? res.json() : res.blob();
  }

  /** What to do about a failed getUserMedia(). */
  function micError(err) {
    const name = err && err.name;
    if (!window.isSecureContext) {
      return 'The microphone only works on HTTPS or http://localhost. Open this page through HTTPS (e.g. your reverse proxy) or an SSH tunnel.';
    }
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      return 'Microphone access is blocked. Allow it in the browser (the icon next to the address bar), then try again.';
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      return 'No microphone was found. Plug one in, or pick another one in Settings → Audio.';
    }
    if (name === 'NotReadableError' || name === 'AbortError') {
      return 'The microphone is in use by another app or tab, or the system blocked it. Close the other app and try again.';
    }
    return `Microphone error: ${(err && err.message) || err}`;
  }

  /** Show an error as a dismissable toast at the bottom of the page. */
  function showError(message) {
    console.error(message);
    const toast = document.createElement('div');
    toast.className = 'smt-toast';
    toast.setAttribute('role', 'alert');
    toast.textContent = message;
    toast.title = 'Click to dismiss';
    toast.addEventListener('click', () => toast.remove());
    document.body.append(toast);
    setTimeout(() => toast.remove(), 10000);
  }

  // Anything nobody caught still gets reported instead of failing silently
  window.addEventListener('unhandledrejection', (e) => {
    showError(e.reason?.message || String(e.reason || 'Something went wrong'));
  });
  window.addEventListener('error', (e) => {
    if (e.message) showError(`Something went wrong on this page: ${e.message}`);
  });

  window.SMT = {
    get, set, serverSettings, openMic, applyOutput, applyDownloads, unlockAudio, keepAwake,
    errorMessage, request, showError, micError,
  };
})();
