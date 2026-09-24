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

  window.SMT = { get, set, serverSettings, openMic, applyOutput, applyDownloads };
})();
