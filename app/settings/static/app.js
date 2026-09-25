/* Settings page: AI connections, assistant, speech recognition, audio. */

const $ = (sel) => document.querySelector(sel);

let providers = [];
let settings = null;
let editingId = null;   // connection being edited (null = new)
let sttModels = [];

const LANGUAGES = [
  ['auto', 'Detect automatically'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'],
  ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'],
  ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'], ['ar', 'Arabic'], ['hi', 'Hindi'],
  ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean'], ['sv', 'Swedish'], ['da', 'Danish'],
  ['no', 'Norwegian'], ['fi', 'Finnish'], ['cs', 'Czech'], ['el', 'Greek'], ['vi', 'Vietnamese'],
];

// ---- API ------------------------------------------------------------------------

async function api(path, options = {}) {
  return SMT.request(path, options);  // throws Error(readable message)
}

const send = (path, method, body) => api(path, {
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

let toastTimer = null;
function toast(text) {
  const el = $('#toast');
  el.textContent = text;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 1800);
}

async function save(values, message = 'Saved') {
  settings = await send('api/settings', 'PUT', values);
  toast(message);
  renderConnections();
  return settings;
}

const providerById = (id) => providers.find((p) => p.id === id) || { label: id, base_url: '' };

// ---- Connections ---------------------------------------------------------------

function renderConnections() {
  const list = $('#conn-list');
  list.innerHTML = '';
  for (const conn of settings.connections) {
    const active = conn.id === settings.activeConnection;
    const row = document.createElement('div');
    row.className = `conn${active ? ' active' : ''}`;

    const main = document.createElement('div');
    main.className = 'conn-main';
    const name = document.createElement('strong');
    name.textContent = conn.name;
    const detail = document.createElement('span');
    detail.textContent = `${providerById(conn.provider).label} · ${conn.model || 'no model chosen'}`;
    main.append(name, detail);
    row.append(main);

    if (active) {
      const pill = document.createElement('span');
      pill.className = 'pill pill--ok';
      pill.textContent = 'Active';
      row.append(pill);
    } else {
      row.append(button('Use', 'btn--secondary', async () => {
        if (!conn.model) return alert('Choose a model for this connection first (Edit).');
        await save({ activeConnection: conn.id }, `Test Lab now uses ${conn.name}`);
        renderActiveSelect();
      }));
    }
    row.append(button('Edit', 'btn--ghost', () => openForm(conn)));
    row.append(button('Delete', 'btn--ghost', async () => {
      if (!confirm(`Delete the connection “${conn.name}”?`)) return;
      await save({ connections: settings.connections.filter((c) => c.id !== conn.id) }, 'Deleted');
      if (editingId === conn.id) closeForm();
      renderActiveSelect();
    }));
    list.append(row);
  }
  renderActiveSelect();
}

function button(text, variant, onClick) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = `btn ${variant}`;
  el.textContent = text;
  el.addEventListener('click', onClick);
  return el;
}

function renderActiveSelect() {
  const select = $('#active-conn');
  select.innerHTML = '';
  select.add(new Option('Fixed reply (no AI)', ''));
  for (const conn of settings.connections) {
    const option = new Option(`${conn.name} · ${conn.model || 'no model'}`, conn.id);
    option.disabled = !conn.model;
    select.add(option);
  }
  select.value = settings.activeConnection || '';
}

$('#active-conn').addEventListener('change', (e) => save({ activeConnection: e.target.value || null }));

function openForm(conn = null) {
  editingId = conn ? conn.id : null;
  $('#conn-form-title').textContent = conn ? `Edit “${conn.name}”` : 'New connection';
  $('#conn-provider').value = conn ? conn.provider : 'openai';
  $('#conn-name').value = conn ? conn.name : '';
  $('#conn-url').value = conn ? conn.baseUrl : '';
  $('#conn-key').value = '';
  $('#conn-model').value = conn ? conn.model : '';
  $('#model-options').innerHTML = '';
  $('#model-count').textContent = '';
  showResult();
  providerChanged(!conn);
  $('#key-hint').textContent = conn && conn.hasKey
    ? `Saved key ${conn.keyHint}. Leave blank to keep it.` : '';
  $('#conn-form').classList.remove('hidden');
  $('#conn-form').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeForm() {
  editingId = null;
  $('#conn-form').classList.add('hidden');
}

function providerChanged(resetUrl) {
  const provider = providerById($('#conn-provider').value);
  const url = $('#conn-url');
  if (resetUrl || !url.value) {
    url.value = provider.base_url;
  }
  url.placeholder = provider.base_url || 'https://your-server/v1';
  $('#provider-note').textContent = provider.note || '';
  $('#conn-key-label').firstChild.textContent = provider.needs_key ? 'API key' : 'API key (optional)';
  const link = $('#key-link');
  link.textContent = provider.key_url ? 'Get a key ↗' : '';
  link.href = provider.key_url || '#';
  $('#load-models-btn').disabled = !provider.lists_models;
  if (!$('#conn-name').value || providers.some((p) => p.label === $('#conn-name').value)) {
    $('#conn-name').value = provider.label;
  }
}

// A different provider means a different endpoint: load its default URL
$('#conn-provider').addEventListener('change', () => {
  $('#model-options').innerHTML = '';
  $('#model-count').textContent = '';
  showResult();
  providerChanged(true);
});
$('#add-conn-btn').addEventListener('click', () => openForm());
$('#cancel-conn-btn').addEventListener('click', closeForm);
$('#toggle-key').addEventListener('click', () => {
  const key = $('#conn-key');
  key.type = key.type === 'password' ? 'text' : 'password';
  $('#toggle-key').textContent = key.type === 'password' ? 'Show' : 'Hide';
});

function draft() {
  return {
    id: editingId || '',
    name: $('#conn-name').value.trim(),
    provider: $('#conn-provider').value,
    baseUrl: $('#conn-url').value.trim(),
    apiKey: $('#conn-key').value.trim(),
    model: $('#conn-model').value.trim(),
  };
}

function showResult(kind = null, text = '') {
  const el = $('#conn-result');
  el.className = `banner${kind ? ` banner--${kind}` : ''}`;
  el.textContent = text;
  el.classList.toggle('hidden', !kind);
}

function fillModels(models) {
  const list = $('#model-options');
  list.innerHTML = '';
  models.forEach((m) => list.append(new Option(m, m)));
  $('#model-count').textContent = models.length
    ? `${models.length} models available: click the box or start typing to pick one.`
    : 'No models listed. Type the model name.';
  if (!$('#conn-model').value && models.length) $('#conn-model').value = models[0];
}

async function withBusy(btn, label, fn) {
  const text = btn.textContent;
  btn.disabled = true;
  btn.textContent = label;
  try {
    await fn();
  } catch (err) {
    showResult('error', err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = text;
  }
}

$('#load-models-btn').addEventListener('click', (e) => withBusy(e.target, 'Loading…', async () => {
  showResult();
  const { models } = await send('api/connections/models', 'POST', draft());
  fillModels(models);
}));

$('#test-conn-btn').addEventListener('click', (e) => withBusy(e.target, 'Testing…', async () => {
  showResult();
  const result = await send('api/connections/test', 'POST', draft());
  if (result.models.length) fillModels(result.models);
  const parts = ['✓ Connected'];
  if (result.models.length) parts.push(`${result.models.length} models`);
  if (result.reply) parts.push(`${$('#conn-model').value} replied “${result.reply}” in ${result.latencyMs} ms`);
  else parts.push('choose a model to test a reply');
  showResult('success', parts.join(' · '));
}));

$('#conn-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const conn = draft();
  const others = settings.connections.filter((c) => c.id !== editingId);
  const connections = editingId
    ? settings.connections.map((c) => (c.id === editingId ? conn : c))
    : [...others, conn];
  try {
    const firstUsable = !settings.activeConnection && conn.model;
    settings = await send('api/settings', 'PUT', { connections });
    if (firstUsable) {
      const saved = editingId ? conn.id : settings.connections[settings.connections.length - 1].id;
      settings = await send('api/settings', 'PUT', { activeConnection: saved });
    }
    toast('Connection saved');
    closeForm();
    renderConnections();
  } catch (err) {
    showResult('error', err.message);
  }
});

// ---- Simple sections (assistant, speech, audio) -------------------------------

function fillSections() {
  document.querySelectorAll('[data-section]').forEach((section) => {
    const values = settings[section.dataset.section];
    section.querySelectorAll('[data-key]').forEach((input) => {
      const value = values[input.dataset.key];
      if (input.type === 'checkbox') input.checked = Boolean(value);
      else input.value = value;
      showValue(section, input);
    });
  });
}

function showValue(section, input) {
  const out = section.querySelector(`[data-value="${input.dataset.key}"]`);
  if (out) out.textContent = input.value;
}

const pending = {};
document.querySelectorAll('[data-section]').forEach((section) => {
  section.querySelectorAll('[data-key]').forEach((input) => {
    const handler = () => {
      showValue(section, input);
      let value = input.value;
      if (input.type === 'checkbox') value = input.checked;
      else if (input.type === 'number' || input.type === 'range') value = Number(input.value);
      const name = section.dataset.section;
      clearTimeout(pending[name + input.dataset.key]);
      pending[name + input.dataset.key] = setTimeout(async () => {
        await save({ [name]: { [input.dataset.key]: value } });
        if (input.dataset.key === 'sttModel') refreshStt();
      }, input.type === 'text' || input.tagName === 'TEXTAREA' ? 600 : 250);
    };
    const live = input.type === 'range' || input.type === 'text' || input.tagName === 'TEXTAREA';
    input.addEventListener(live ? 'input' : 'change', handler);
  });
});

// ---- Speech recognition --------------------------------------------------------

async function refreshStt() {
  const status = await api(`api/stt?model=${encodeURIComponent(settings.speech.sttModel)}`);
  const pill = $('#stt-status');
  if (!status.available) {
    pill.className = 'pill pill--bad';
    pill.textContent = 'Not installed';
  } else if (status.downloading) {
    pill.className = 'pill pill--warn pill--live';
    pill.textContent = 'Downloading…';
  } else if (status.cached) {
    pill.className = 'pill pill--ok';
    pill.textContent = 'Downloaded';
  } else {
    pill.className = 'pill pill--warn';
    pill.textContent = 'Not downloaded yet';
  }
  $('#stt-download-btn').disabled = !status.available || status.cached || status.downloading;
  return status;
}

$('#stt-download-btn').addEventListener('click', async (e) => {
  const pill = $('#stt-status');
  pill.className = 'pill pill--warn pill--live';
  pill.textContent = 'Downloading…';
  e.target.disabled = true;
  try {
    await send('api/stt/download', 'POST', { model: settings.speech.sttModel });
    toast('Speech model ready');
  } catch (err) {
    SMT.showError(err.message);
  }
  refreshStt();
});

// ---- Audio devices ---------------------------------------------------------------

let testStream = null;
let testRaf = null;

async function listDevices() {
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (e) {
    return;
  }
  fillDeviceSelect($('#mic-select'), devices.filter((d) => d.kind === 'audioinput'),
    'Default microphone', SMT.get('input'));
  fillDeviceSelect($('#speaker-select'), devices.filter((d) => d.kind === 'audiooutput'),
    'Default speaker', SMT.get('output'));
}

function fillDeviceSelect(select, devices, defaultLabel, wanted) {
  select.innerHTML = '';
  select.add(new Option(defaultLabel, ''));
  devices
    .filter((d) => d.deviceId && d.deviceId !== 'default' && d.deviceId !== 'communications')
    .forEach((d, i) => select.add(new Option(d.label || `${defaultLabel.split(' ')[1]} ${i + 1}`, d.deviceId)));
  select.value = [...select.options].some((o) => o.value === wanted) ? wanted : '';
}

$('#mic-select').addEventListener('change', (e) => {
  SMT.set('input', e.target.value);
  toast('Microphone saved for all tabs');
  if (testStream) { stopMicTest(); startMicTest(); }
});
$('#speaker-select').addEventListener('change', (e) => {
  SMT.set('output', e.target.value);
  toast('Speaker saved for all tabs');
});
$('#volume').addEventListener('input', (e) => {
  SMT.set('volume', e.target.value);
  $('#volume-value').textContent = `${Math.round(e.target.value * 100)}%`;
});

async function startMicTest() {
  try {
    testStream = await SMT.openMic({
      echoCancellation: settings.audio.echoCancellation,
      noiseSuppression: settings.audio.noiseSuppression,
      autoGainControl: settings.audio.autoGainControl,
    }, $('#mic-select').value);
  } catch (err) {
    SMT.showError(SMT.micError(err));
    return;
  }
  await listDevices();  // names are visible once the microphone is allowed
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(testStream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  const tick = () => {
    if (!testStream) { ctx.close(); return; }
    analyser.getFloatTimeDomainData(samples);
    let peak = 0;
    for (const s of samples) peak = Math.max(peak, Math.abs(s));
    const bar = $('#vu-bar');
    bar.style.width = `${Math.min(100, peak * 100)}%`;
    bar.classList.toggle('hot', peak > 0.6);
    bar.classList.toggle('clip', peak > 0.95);
    testRaf = requestAnimationFrame(tick);
  };
  tick();
  $('#mic-test-btn').textContent = '⏹ Stop';
}

function stopMicTest() {
  if (testStream) testStream.getTracks().forEach((t) => t.stop());
  testStream = null;
  cancelAnimationFrame(testRaf);
  $('#vu-bar').style.width = '0';
  $('#mic-test-btn').textContent = '🎤 Test';
}

$('#mic-test-btn').addEventListener('click', () => (testStream ? stopMicTest() : startMicTest()));

// A short two-tone chime as a WAV blob, so it plays through an <audio> element
// (and therefore through the chosen speaker).
function chimeWav() {
  const rate = 22050;
  const n = Math.floor(rate * 0.7);
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const freq = t < 0.3 ? 660 : 880;
    const env = Math.min(1, i / 300) * Math.min(1, (n - i) / 3000);
    v.setInt16(44 + i * 2, Math.sin(2 * Math.PI * freq * t) * 0.4 * env * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

$('#speaker-test-btn').addEventListener('click', async () => {
  const audio = new Audio(URL.createObjectURL(chimeWav()));
  await SMT.applyOutput(audio);
  audio.play();
});

// ---- Init --------------------------------------------------------------------------

async function init() {
  const [catalog, current, stt] = await Promise.all([
    api('api/providers'),
    api('api/settings'),
    api('api/stt'),
  ]);
  providers = catalog.providers;
  settings = current;
  sttModels = stt.models;

  providers.forEach((p) => $('#conn-provider').add(new Option(p.label, p.id)));
  sttModels.forEach((m) => $('#stt-model').add(new Option(m.label, m.id)));
  LANGUAGES.forEach(([code, name]) => $('#stt-language').add(new Option(name, code)));

  renderConnections();
  fillSections();
  refreshStt();

  const volume = SMT.get('volume', '1');
  $('#volume').value = volume;
  $('#volume-value').textContent = `${Math.round(volume * 100)}%`;
  if (!('setSinkId' in HTMLMediaElement.prototype)) {
    $('#speaker-select').disabled = true;
    $('#speaker-note').classList.remove('hidden');
  }
  listDevices();
  navigator.mediaDevices?.addEventListener('devicechange', listDevices);
}

init();

// Refresh download status and devices when the tab is shown again
window.addEventListener('message', (e) => {
  if (e.origin === location.origin && e.data?.type === 'smt:shown' && settings) {
    refreshStt();
    listDevices();
  }
});
