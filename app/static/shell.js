/* Tab switching (hash routing) and the status shown on the Home tab. */

const TABS = ['home', 'wakeword', 'voice', 'lab', 'settings'];

function showTab(name) {
  if (!TABS.includes(name)) name = 'home';

  for (const tab of TABS) {
    const panel = document.getElementById(`panel-${tab}`);
    panel.classList.toggle('active', tab === name);

    // Create each tool's iframe on first visit, then keep it alive
    if (tab === name && panel.dataset.src) {
      const existing = panel.querySelector('iframe');
      if (existing) {
        // Let the page refresh its choices (new models, changed settings)
        existing.contentWindow.postMessage({ type: 'smt:shown' }, location.origin);
      } else {
        const frame = document.createElement('iframe');
        frame.src = panel.dataset.src;
        frame.title = tab;
        frame.allow = 'microphone; autoplay; clipboard-write; speaker-selection';
        panel.appendChild(frame);
      }
    }
  }

  document.querySelectorAll('.tab').forEach((el) => {
    const active = el.dataset.tab === name;
    el.classList.toggle('active', active);
    el.setAttribute('aria-selected', active);
  });
}

window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
showTab(location.hash.slice(1));

// ---- Status -----------------------------------------------------------------

function set(id, text, cls = '') {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = cls;
  el.title = text;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

async function refreshStatus() {
  let status;
  try {
    const res = await fetch('api/status');
    if (!res.ok) throw new Error(res.statusText);
    status = await res.json();
  } catch {
    return;
  }

  const ww = status.wakeword;
  if (ww.dataReady) set('ww-data', 'Ready', 'ok');
  else if (ww.preparing) set('ww-data', 'Downloading…', 'warn');
  else set('ww-data', 'Missing', 'warn');
  set('ww-models', plural(ww.models.length, 'model'));
  set('ww-now', ww.training ? `Training “${ww.training}”` : 'Idle', ww.training ? 'warn' : '');

  const v = status.voice;
  if (!v.device.ok) set('v-device', 'Missing', 'bad');
  else if (v.device.gpu) set('v-device', `${v.device.gpu} (${v.device.memory_gb} GB)`, 'ok');
  else set('v-device', 'CPU only (slow)', 'warn');
  set('v-voices', plural(v.voices.length, 'voice'));
  set('v-now', v.training ? `Training “${v.training}”` : 'Idle', v.training ? 'warn' : '');

  const wwBusy = Boolean(ww.training || ww.preparing);
  document.getElementById('dot-wakeword').classList.toggle('busy', wwBusy);
  document.getElementById('dot-voice').classList.toggle('busy', Boolean(v.training));
  const a = status.assistant;
  set('lab-mode', a.connection ? 'AI answers' : 'Fixed reply', a.connection ? 'ok' : '');
  set('lab-ai', a.connection ? `${a.provider} · ${a.model}` : 'None', '');
  set('lab-ready', ww.models.length ? 'Ready' : 'Train a wake word', ww.models.length ? 'ok' : 'warn');

  document.getElementById('gpu-warning').classList.toggle(
    'hidden', !(ww.training && v.training && v.device.gpu)
  );
}

refreshStatus();
setInterval(refreshStatus, 5000);
