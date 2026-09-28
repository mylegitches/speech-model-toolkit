'use strict';

const $ = (sel) => document.querySelector(sel);
const show = (el, visible) => el.classList.toggle('hidden', !visible);

const TEST_SENTENCES = {
  en: 'Hello! This is my new voice. The front door is locked and the lights are off.',
};

let info = null;          // /api/info
let voice = null;         // selected voice (from /api/voices/{name})
let status = null;        // latest training status
let events = null;        // EventSource
let logCount = 0;
let selectedPreset = null;

async function api(path, options = {}) {
  return SMT.request(path, options);  // throws Error(readable message)
}

function postJson(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

function voiceUrl(suffix = '') {
  return `api/voices/${encodeURIComponent(voice.name)}${suffix}`;
}

// ---------------------------------------------------------------------------
// 1. Voice

// Which tab this page is (set in index.html from ?view=): dataset, train or voices
const VIEW = document.documentElement.dataset.view || 'dataset';
const hasModel = (v) => Boolean(v.training.exports.length);

/** The choices for this tab: datasets, or (Voices) the ones with a trained model. */
async function refreshVoiceList() {
  const { voices } = await api('api/voices');
  const list = VIEW === 'voices' ? voices.filter(hasModel) : voices;
  const select = $('#voice-select');
  const current = select.value;
  select.innerHTML = '';
  select.add(new Option(VIEW === 'voices' ? '— choose a voice model —' : '— choose a dataset —', ''));
  list.forEach((v) => {
    const latest = v.training.exports[0];
    const label = {
      dataset: `${v.name} · ${v.languageName} · ${v.recorded} clips`,
      train: `${v.name} · ${v.recorded} clips${hasModel(v) ? ` · trained to epoch ${latest.epoch}` : ''}`,
      voices: `${v.name} · ${v.languageName} · ${v.training.exports.length} version${v.training.exports.length === 1 ? '' : 's'}, latest epoch ${latest?.epoch}`,
    }[VIEW];
    select.add(new Option(label, v.name));
  });
  if (list.some((v) => v.name === current)) select.value = current;
  show($('#no-models'), VIEW === 'voices' && !list.length);
  return list;
}

async function loadVoices(selectName) {
  const list = await refreshVoiceList();
  const select = $('#voice-select');
  let name = selectName;
  if (!name) {
    try { name = localStorage.getItem('voice'); } catch (e) { name = null; }
  }
  if (name && list.some((v) => v.name === name)) {
    select.value = name;
  } else if (list.length > 0) {
    select.value = list[0].name;
  }

  show($('#new-voice-form'), VIEW === 'dataset' && list.length === 0);
  await selectVoice(select.value);
}

/** Remember the choice for the other tabs (only choices you make, so tabs don't pull each other around). */
function rememberVoice(name) {
  try { localStorage.setItem('voice', name || ''); } catch (e) { /* ignore */ }
}

// Another tab picked a dataset or voice: follow it, if it's one this tab lists
window.addEventListener('storage', async (e) => {
  if (e.key !== 'voice' || !e.newValue || e.newValue === voice?.name) return;
  const list = await refreshVoiceList();
  if (list.some((v) => v.name === e.newValue)) {
    $('#voice-select').value = e.newValue;
    selectVoice(e.newValue);
  }
});

// Back on this tab: new datasets or models may have appeared (the current work stays as it is)
window.addEventListener('message', async (e) => {
  if (e.origin !== location.origin || e.data?.type !== 'smt:shown') return;
  const list = await refreshVoiceList();
  if (!voice || !list.some((v) => v.name === voice.name)) {
    if (list.length) {
      $('#voice-select').value = list[0].name;
      selectVoice(list[0].name);
    } else if (voice) {
      selectVoice('');
    }
  } else if (VIEW !== 'dataset') {
    selectVoice(voice.name);  // Train / Voices: fresh training state and versions
  }
});

async function selectVoice(name) {
  if (events) {
    events.close();
    events = null;
  }
  voice = name ? await api(`api/voices/${encodeURIComponent(name)}`) : null;

  [$('#record-card'), $('#manage-card'), $('#train-card'), $('#test-card')].forEach((card) => show(card, !!voice));
  if (!voice) {
    $('#voice-summary').textContent = '';
    show($('#next-train'), false);
    return;
  }

  $('#voice-summary').textContent =
    `${voice.languageName} · ${voice.gender} · phonemes: ${voice.espeak_voice} · model: ${voice.modelName}.onnx`;
  updateRecorded(voice.recorded);
  logCount = 0;
  $('#log-panel').textContent = '';

  if (VIEW === 'dataset') {
    charUi.name = null;
    show($('#character-panel'), false);
    skip = 0;
    $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
    $('#record-status').textContent = mediaStream
      ? 'Read each sentence naturally in a quiet room. Press R to record, R again to stop.'
      : 'First pick your microphone and click “Enable microphone”. Use the same one every session.';
    await loadPrompt();
    if (micAvailable()) {
      await listMicrophones();
      checkMicMatch();
    }
    loadManager();
  }
  if (VIEW === 'train') {
    matchState = null;
    fillTrainForm(voice.defaults);
    loadMatch();
  }
  if (VIEW === 'voices') {
    loadSpeed();
    $('#speak-input').value = TEST_SENTENCES[voice.language.split('-')[0]] || '';
  }
  // Files waiting for review (Build Dataset lists them; Train warns about them)
  $('#free-takes').innerHTML = '';
  Object.keys(renderedTakes).forEach((id) => delete renderedTakes[id]);
  if (VIEW !== 'voices') loadTakes();
  renderStatus(voice.training);
  if (VIEW !== 'dataset') {
    refreshInfo();  // is another voice training right now?
    connectEvents();
  }
}

$('#voice-select').addEventListener('change', (e) => {
  rememberVoice(e.target.value);
  selectVoice(e.target.value);
});

// "Next" links: the next tab opens on this dataset / voice
['#next-train', '#next-voices', '#next-lab'].forEach((id) => $(id).addEventListener('click', () => {
  rememberVoice(voice?.name);
  if (id === '#next-lab') {
    try { localStorage.setItem('smt.lab.voice', `voice:${voice.name}`); } catch (e) { /* ignore */ }
  }
}));
$('#new-voice-btn').addEventListener('click', () => {
  show($('#new-voice-form'), true);
  $('#new-name').focus();
});
$('#cancel-new-voice').addEventListener('click', () => show($('#new-voice-form'), false));

$('#new-voice-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const created = await postJson('api/voices', {
      name: $('#new-name').value,
      language: $('#new-language').value,
      gender: $('#new-gender').value,
    });
    $('#new-name').value = '';
    rememberVoice(created.name);
    await loadVoices(created.name);
  } catch (err) {
    SMT.showError(err.message);
  }
});

// ---------------------------------------------------------------------------
// 2. Record

let prompt = null;
let skip = 0;
let mediaStream = null;
let recorder = null;
let chunks = [];
let recordedBlob = null;
let analyser = null;

function updateRecorded(count) {
  voice.recorded = count;
  $('#recorded-count').textContent = count;
  show($('#next-train'), count >= 50);  // enough to train: on to the next tab
  // Clips added or removed elsewhere on the page: the list of clips follows (unless you're editing it)
  if (VIEW === 'dataset' && count !== manageUi.clips.length && !manageUi.edits.size && !manageUi.deleted.size) {
    clearTimeout(manageUi.timer);
    manageUi.timer = setTimeout(loadManager, 800);
  }
  const exportLink = $('#export-dataset');
  exportLink.href = voiceUrl('/dataset.zip');
  exportLink.classList.toggle('hidden', !count);
  const fill = $('#readiness-fill');
  fill.style.width = `${Math.min(100, count / 10)}%`;
  let text;
  if (count < 50) {
    text = `at least 50 clips needed to train (${50 - count} to go)`;
    fill.style.background = 'var(--danger)';
  } else if (count < 300) {
    text = 'enough to train · 300+ sounds much better';
    fill.style.background = 'var(--warning)';
  } else {
    text = 'great · more recordings still help';
    fill.style.background = 'var(--success)';
  }
  $('#readiness-text').textContent = text;
  renderStartOptions();  // Find match needs a few recordings
}

async function loadPrompt() {
  const result = await api(voiceUrl(`/prompt?skip=${skip}`));
  prompt = result.prompt;
  updateRecorded(result.recorded);
  resetTake();
  if (prompt) {
    $('#prompt-text').textContent = prompt.text;
  } else {
    $('#prompt-text').textContent = skip > 0
      ? 'No more sentences after the skipped ones. Reload to see them again.'
      : 'All sentences recorded — nice work!';
  }
  updateRecordButton();
  $('#skip-btn').disabled = !prompt;
}

function resetTake() {
  recordedBlob = null;
  $('#play-btn').disabled = true;
  $('#save-btn').disabled = true;
  $('#prompt-box').classList.remove('recording', 'recorded');
}

let micLabel = '';          // label of the microphone currently open

function micAvailable() {
  return window.isSecureContext && !!navigator.mediaDevices;
}

function updateRecordButton() {
  $('#record-btn').disabled = !prompt || !mediaStream;
  $('#record-btn').title = mediaStream ? '' : 'Enable your microphone first';
}

async function listMicrophones() {
  const select = $('#mic-select');
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === 'audioinput');
    const named = devices.filter((d) => d.deviceId && d.label);
    if (named.length === 0) {
      // Browsers hide device names until the microphone is allowed once
      select.innerHTML = '<option value="">Click “Enable microphone” to list your microphones</option>';
      select.disabled = true;
      return;
    }

    const current = select.value;
    select.innerHTML = '';
    named.forEach((d) => select.add(new Option(d.label, d.deviceId)));
    select.disabled = false;

    let wanted = current;
    if (!wanted) {
      try { wanted = localStorage.getItem('mic') || SMT.get('input'); } catch (e) { wanted = ''; }
    }
    const voiceMic = voice && voice.microphone && named.find((d) => d.label === voice.microphone);
    if (voiceMic && !mediaStream) {
      wanted = voiceMic.deviceId;  // default to what this voice was recorded with
    }
    select.value = named.some((d) => d.deviceId === wanted) ? wanted : named[0].deviceId;
  } catch (e) {
    select.innerHTML = '<option value="">No microphones found</option>';
  }
}

async function openMicrophone() {
  if (mediaStream) {
    return mediaStream;
  }
  const deviceId = $('#mic-select').value;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      // Raw audio is best for training
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });
  micLabel = mediaStream.getAudioTracks()[0].label || '';
  const context = new AudioContext();
  context.resume().catch(() => {});  // phones may create it suspended (level meter only)
  analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  context.createMediaStreamSource(mediaStream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  const tick = () => {
    if (!analyser) {
      return;
    }
    analyser.getFloatTimeDomainData(samples);
    let peak = 0;
    for (const s of samples) {
      peak = Math.max(peak, Math.abs(s));
    }
    $('#vu-bar').style.width = `${Math.min(100, peak * 100)}%`;
    $('#vu-bar').style.background = peak > 0.95 ? 'var(--danger)' : 'var(--success)';
    requestAnimationFrame(tick);
  };
  tick();
  await listMicrophones();
  // Show the device actually opened
  const opened = Array.from($('#mic-select').options).find((o) => o.text === micLabel);
  if (opened) {
    $('#mic-select').value = opened.value;
  }
  $('#mic-test-btn').textContent = '🎤 Microphone on';
  updateRecordButton();
  checkMicMatch();
  return mediaStream;
}

function closeMicrophone() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    analyser = null;
    micLabel = '';
    $('#vu-bar').style.width = '0';
    $('#mic-test-btn').textContent = '🎤 Enable microphone';
  }
  updateRecordButton();
}

async function enableMicrophone() {
  try {
    await openMicrophone();
    $('#record-status').textContent = 'Microphone on. Say something and watch the green bar (red means too loud), then press R to record.';
  } catch (err) {
    showMicProblem(SMT.micError(err));
  }
}

function showMicProblem(text) {
  $('#mic-warning').textContent = text;
  show($('#mic-warning'), !!text);
}

function checkMicMatch() {
  if (voice && voice.microphone && micLabel && micLabel !== voice.microphone) {
    showMicProblem(`This voice was recorded with “${voice.microphone}”, but “${micLabel}” is selected. `
      + 'Switch back so every clip sounds the same.');
  } else {
    showMicProblem('');
  }
}

$('#mic-select').addEventListener('change', () => {
  closeMicrophone();
  try { localStorage.setItem('mic', $('#mic-select').value); } catch (e) { /* ignore */ }
  enableMicrophone();
});
$('#mic-test-btn').addEventListener('click', () => {
  if (!mediaStream) {
    enableMicrophone();
  }
});

function preferredMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
}

async function toggleRecording() {
  if (!prompt) {
    return;
  }
  if (recorder && recorder.state === 'recording') {
    // Keep the tail of the last word
    $('#record-btn').disabled = true;
    setTimeout(() => recorder.stop(), 400);
    return;
  }

  if (!mediaStream) {
    $('#record-status').textContent = 'Pick your microphone and click “Enable microphone” first.';
    return;
  }

  chunks = [];
  const mimeType = preferredMimeType();
  recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    recordedBlob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    $('#playback').src = URL.createObjectURL(recordedBlob);
    $('#record-btn').classList.remove('recording');
    $('#record-btn').innerHTML = '● Re-record <kbd>R</kbd>';
    $('#record-btn').disabled = false;
    $('#play-btn').disabled = false;
    $('#save-btn').disabled = false;
    $('#prompt-box').classList.replace('recording', 'recorded');
    $('#record-status').textContent = 'Listen back with P, then save with S (or re-record with R).';
  };
  recorder.start(250);
  resetTake();
  $('#prompt-box').classList.add('recording');
  $('#record-btn').classList.add('recording');
  $('#record-btn').innerHTML = '■ Stop <kbd>R</kbd>';
  $('#record-status').textContent = 'Recording… read the sentence, then press R.';
}

async function saveRecording() {
  if (!recordedBlob || !prompt) {
    return;
  }
  const form = new FormData();
  form.set('group', prompt.group);
  form.set('id', prompt.id);
  form.set('text', prompt.text);
  form.set('audio', recordedBlob, 'audio');
  form.set('mic', micLabel);
  $('#save-btn').disabled = true;
  try {
    const result = await api(voiceUrl('/recordings'), { method: 'POST', body: form });
    updateRecorded(result.recorded);
    voice.microphone = result.microphone;
    checkMicMatch();
    $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
    $('#record-status').textContent = 'Saved. Next sentence:';
    await loadPrompt();
  } catch (err) {
    $('#save-btn').disabled = false;
    $('#record-status').textContent = `Save failed: ${err.message}`;
  }
}

$('#record-btn').addEventListener('click', toggleRecording);
$('#play-btn').addEventListener('click', async () => {
  await SMT.applyOutput($('#playback'));
  $('#playback').play();
});
$('#save-btn').addEventListener('click', saveRecording);
$('#skip-btn').addEventListener('click', async () => {
  skip += 1;
  $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
  await loadPrompt();
});

document.addEventListener('keydown', (e) => {
  if (!voice || e.ctrlKey || e.metaKey || e.altKey || e.repeat) {
    return;
  }
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    return;
  }
  if (recordMode !== 'prompts') {
    return;  // shortcuts are for reading prompts
  }
  if (player.box && !player.box.classList.contains('hidden')) {
    return;  // the play-through has its own keys
  }
  const key = e.key.toLowerCase();
  const actions = { r: '#record-btn', p: '#play-btn', s: '#save-btn', k: '#skip-btn' };
  if (actions[key] && !$(actions[key]).disabled) {
    e.preventDefault();
    $(actions[key]).click();
  }
});

// ---- Freeform: record freely (or upload), transcribe, review clips ------------------

const FREE_MAX_SECONDS = 30 * 60;
let recordMode = 'prompts';
let freeRecorder = null;
let freeChunks = [];
let freeTimer = null;
let freeStarted = 0;
let takesTimer = null;
const renderedTakes = {};  // take id -> state it was last rendered in
const takeEdits = {};      // take id -> {clips: {index: {keep, text}}, speakers: Set}

function setMode(mode) {
  if (!['prompts', 'free', 'file', 'clone'].includes(mode)) mode = 'prompts';
  recordMode = mode;
  document.querySelectorAll('.mode-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  show($('#mic-block'), mode === 'prompts' || mode === 'free');
  show($('#prompt-mode'), mode === 'prompts');
  show($('#free-mode'), mode === 'free');
  show($('#file-mode'), mode === 'file');
  show($('#clone-mode'), mode === 'clone');
  show($('#takes-block'), mode !== 'prompts');  // takes waiting for review
  try { localStorage.setItem('voice.recordMode', mode); } catch (e) { /* ignore */ }
  setTimeout(modeViews, 0);  // after the page script has loaded (setMode also runs while it loads)
}

/** Each tab shows its own files: Character clone files there, the others under Speak freely / Import file. */
function modeViews() {
  if (!voice) return;
  applyTakeFilter();
  renderPeople();
  renderClone();
}
document.querySelectorAll('.mode-btn').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
try { setMode(localStorage.getItem('voice.recordMode') || 'prompts'); } catch (e) { setMode('prompts'); }

function clock(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` : `${m}:${String(s % 60).padStart(2, '0')}`;
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

$('#free-record-btn').addEventListener('click', async () => {
  const button = $('#free-record-btn');
  if (freeRecorder && freeRecorder.state === 'recording') {
    button.disabled = true;
    setTimeout(() => freeRecorder.stop(), 400);  // keep the last word
    return;
  }
  try {
    await openMicrophone();
  } catch (err) {
    SMT.showError(SMT.micError(err));
    return;
  }
  freeChunks = [];
  const mimeType = preferredMimeType();
  freeRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  freeRecorder.ondataavailable = (e) => freeChunks.push(e.data);
  freeRecorder.onstop = async () => {
    clearInterval(freeTimer);
    SMT.keepAwake(false);
    const type = freeRecorder.mimeType || 'audio/webm';
    const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
    button.classList.remove('recording');
    button.textContent = '● Start recording';
    button.disabled = false;
    show($('#free-timer'), false);
    await uploadTake(new Blob(freeChunks, { type }), `take.${ext}`, false, $('#free-denoise').value);
  };
  freeRecorder.start(1000);
  freeStarted = Date.now();
  SMT.keepAwake(true);
  button.classList.add('recording');
  button.textContent = '■ Stop and transcribe';
  show($('#free-timer'), true);
  $('#free-timer').textContent = '0:00';
  freeTimer = setInterval(() => {
    const elapsed = (Date.now() - freeStarted) / 1000;
    $('#free-timer').textContent = clock(elapsed);
    if (elapsed >= FREE_MAX_SECONDS) button.click();
  }, 500);
});

// ---- Import: files or whole folders (with subfolders) ----------------------------

const MEDIA_FILE = /\.(mkv|mp4|m4v|avi|mov|wmv|flv|webm|ts|m2ts|mts|mpg|mpeg|vob|3gp|ogv|mp3|wav|flac|ogg|opus|m4a|m4b|aac|ac3|eac3|dts|wma|aiff?)$/i;
const VIDEO_FILE = /\.(mkv|mp4|m4v|avi|mov|wmv|flv|webm|ts|m2ts|mts|mpg|mpeg|vob|3gp|ogv)$/i;
function formatSize(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

// Files and folders dropped on the page
async function readEntry(entry, prefix = '') {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    return [{ file, path: prefix + file.name }];
  }
  if (!entry.isDirectory) return [];
  const reader = entry.createReader();
  const children = [];
  // readEntries returns results in batches until it returns an empty list
  for (;;) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    children.push(...batch);
  }
  const nested = await Promise.all(children.map((c) => readEntry(c, `${prefix}${entry.name}/`)));
  return nested.flat();
}

/**
 * A file picker + list + upload button (Import file, and Character clone).
 * Files and whole folders (with subfolders) can be picked or dropped; the list
 * shows them with checkboxes, and the button uploads the ticked ones in order.
 */
function makeImporter({ list, button, file, fileBtn, folder, folderBtn, drop, status, options, picked, label }) {
  const state = { files: [], skipped: 0 };

  function set(entries) {
    const media = entries.filter((e) => MEDIA_FILE.test(e.path) || e.file.type.startsWith('video/') || e.file.type.startsWith('audio/'));
    state.skipped = entries.length - media.length;
    // Adding to what's listed (e.g. a second season), without duplicates
    const known = new Set(state.files.map((e) => e.path));
    state.files = [...state.files, ...media.filter((e) => !known.has(e.path)).map((e) => ({ ...e, selected: true }))]
      .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
    if (picked) picked(state.files);
    render();
  }

  function render() {
    const box = $(list);
    box.innerHTML = '';
    const chosen = state.files.filter((e) => e.selected);
    $(button).textContent = label(chosen.length);
    $(button).disabled = chosen.length === 0;
    if (!state.files.length) {
      if (state.skipped) box.append(el('p', { className: 'hint', textContent: `No audio or video files found (${state.skipped} other files skipped).` }));
      show(box, state.skipped > 0);
      return;
    }
    show(box, true);
    const total = chosen.reduce((sum, e) => sum + e.file.size, 0);
    const all = el('input', { type: 'checkbox', checked: chosen.length === state.files.length, title: 'Select all' });
    all.addEventListener('change', () => { state.files.forEach((e) => { e.selected = all.checked; }); render(); });
    const clear = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Clear list' });
    clear.addEventListener('click', () => { state.files = []; state.skipped = 0; render(); });
    box.append(el('div', { className: 'import-head' }, all, el('strong', {
      textContent: `${chosen.length} of ${state.files.length} file${state.files.length === 1 ? '' : 's'} · ${formatSize(total)}`
        + (state.skipped ? ` · ${state.skipped} other file${state.skipped === 1 ? '' : 's'} skipped` : ''),
    }), clear));
    const rows = el('div', { className: 'import-files' });
    state.files.forEach((entry) => {
      const check = el('input', { type: 'checkbox', checked: entry.selected });
      check.addEventListener('change', () => { entry.selected = check.checked; render(); });
      rows.append(el('label', { className: 'import-file' }, check,
        el('span', { className: 'import-path', textContent: entry.path, title: entry.path }),
        el('span', { className: 'dur', textContent: formatSize(entry.file.size) })));
    });
    box.append(rows);
  }

  $(file).addEventListener('change', () => {
    set([...$(file).files].map((f) => ({ file: f, path: f.name })));
    $(file).value = '';
  });
  if (fileBtn) $(fileBtn).addEventListener('click', () => $(file).click());
  $(folderBtn).addEventListener('click', () => $(folder).click());
  $(folder).addEventListener('change', () => {
    // webkitRelativePath keeps the folder structure: "The Sopranos/Season 1/episode 03.mkv"
    set([...$(folder).files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name })));
    $(folder).value = '';
  });

  const zone = $(drop);
  ['dragenter', 'dragover'].forEach((type) => zone.addEventListener(type, (e) => {
    e.preventDefault();
    zone.classList.add('dropping');
  }));
  ['dragleave', 'drop'].forEach((type) => zone.addEventListener(type, () => zone.classList.remove('dropping')));
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    const items = [...(e.dataTransfer.items || [])];
    try {
      const entries = items.map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
      const found = entries.length
        ? (await Promise.all(entries.map((entry) => readEntry(entry)))).flat()
        : [...e.dataTransfer.files].map((f) => ({ file: f, path: f.name }));
      set(found);
    } catch (err) {
      SMT.showError(`Could not read the dropped files: ${err.message}`);
    }
  });

  $(button).addEventListener('click', async () => {
    const chosen = state.files.filter((e) => e.selected);
    if (!chosen.length) return;
    const go = $(button);
    go.disabled = true;
    const { diarize, denoise, clone } = options();
    let failed = 0;
    for (const [i, entry] of chosen.entries()) {
      const what = chosen.length > 1 ? `${i + 1} of ${chosen.length}: ${entry.path}` : entry.path;
      const ok = await uploadTake(entry.file, entry.file.name, diarize, denoise, entry.path, what, $(status), clone);
      if (ok) entry.selected = false; else failed += 1;
      render();
      go.disabled = true;
    }
    state.files = state.files.filter((e) => e.selected);  // keep the failed ones to retry
    render();
    $(status).textContent = failed
      ? `${failed} file${failed === 1 ? '' : 's'} could not be uploaded (still listed above to retry).`
      : (chosen.length > 1 ? `All ${chosen.length} files uploaded. They're processed one at a time.` : '');
  });

  return { set, render, state };
}

makeImporter({
  list: '#import-list', button: '#free-upload-btn', file: '#free-file', folder: '#free-folder',
  folderBtn: '#free-folder-btn', drop: '#file-mode', status: '#free-upload-status',
  label: (n) => (n > 1 ? `Import ${n} files` : 'Import'),
  options: () => ({ diarize: $('#free-diarize').checked, denoise: $('#file-denoise').value }),
  picked: (files) => {
    // Videos usually have several speakers (and a soundtrack)
    if (files.some((e) => VIDEO_FILE.test(e.path) || e.file.type.startsWith('video/'))) {
      $('#free-diarize').checked = true;
      $('#file-denoise').value = 'strong';
    }
  },
});

// Character clone: always several speakers; the AI then says who says what
makeImporter({
  list: '#clone-list', button: '#clone-upload-btn', file: '#clone-file', fileBtn: '#clone-file-btn', folder: '#clone-folder',
  folderBtn: '#clone-folder-btn', drop: '#clone-mode', status: '#clone-status',
  label: (n) => (n > 1 ? `Add ${n} files` : 'Add'),
  options: () => ({ diarize: true, denoise: $('#clone-denoise').value, clone: true }),
});

function uploadTake(blob, filename, diarize, denoise, originalName = '', label = '', status = $('#free-upload-status'), clone = false) {
  const form = new FormData();
  form.set('audio', blob, filename);
  form.set('denoise', denoise);
  form.set('diarize', diarize ? 'true' : 'false');
  form.set('mic', micLabel || '');
  form.set('original_name', originalName);
  form.set('clone', clone ? 'true' : 'false');
  const what = label ? `Uploading ${label}` : 'Uploading';
  status.textContent = label ? `${what}…` : '';
  // XHR (not fetch) for upload progress: a movie can take a while to send
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', voiceUrl('/freeform'));
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 5e6) {
        status.textContent = `${what}… ${Math.round((e.loaded / e.total) * 100)}% of ${formatSize(e.total)}`;
      }
    };
    xhr.onload = async () => {
      status.textContent = '';
      let ok = true;
      if (xhr.status >= 400) {
        ok = false;
        const message = await SMT.errorMessage(new Response(xhr.responseText, { status: xhr.status, statusText: xhr.statusText }));
        SMT.showError(`Could not import ${originalName || filename}: ${message}`);
      }
      loadTakes();
      resolve(ok);
    };
    xhr.onerror = () => {
      status.textContent = '';
      SMT.showError(`Upload of ${originalName || filename} failed: the connection dropped. Check your network, and your reverse proxy’s upload size limit and timeouts.`);
      resolve(false);
    };
    xhr.send(form);
  });
}

// ---- People recognised across files ------------------------------------------

let people = { people: [], target: null };
let peopleTimer = null;

/** Keep the AI status from GET speakers when an update returns only the people. */
function setPeople(data) {
  people = { ...data, identify: data.identify || people.identify };
}
let lastTakes = [];
const peopleUi = { showAll: false, filter: '', selected: new Set(), showHidden: false };
const PEOPLE_SHOWN = 12;  // more than this: the rest behind "Show all"

const personById = (id) => people.people.find((p) => p.id === id);

function displayName(speaker) {
  const person = speaker.person && personById(speaker.person);
  return person ? person.name : speakerName(speaker.id);
}

async function loadPeople() {
  try {
    people = await api(voiceUrl('/speakers'));
  } catch (err) {
    people = { people: [], target: null };
  }
  try {
    peopleUi.characters = (await api(voiceUrl('/characters'))).characters;
  } catch (err) {
    peopleUi.characters = [];
  }
  // Poll while the AI is identifying people
  clearTimeout(peopleTimer);
  if (people.identify?.running) peopleTimer = setTimeout(() => { if (voice) loadPeople(); }, 3000);
  const ids = new Set(people.people.map((p) => p.id));
  peopleUi.selected.forEach((id) => { if (!ids.has(id)) peopleUi.selected.delete(id); });
  renderPeople();
  renderClone();
  // New files read by the AI: the open character's review picks up their clips (ticks and edits stay)
  const current = (peopleUi.characters || []).find((c) => c.name === charUi.name);
  const sig = current ? `${current.confirmed}/${current.possible}/${current.files}` : '';
  if (charUi.name && charUi.data && current && charUi.sig && sig !== charUi.sig) openCharacter(charUi.name, true);
  if (current) charUi.sig = sig;
}

async function updatePerson(person, change) {
  try {
    setPeople(await api(voiceUrl(`/speakers/${person.id}`), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change),
    }));
  } catch (err) {
    SMT.showError(err.message);
    return;
  }
  if ('target' in change) selectTargetEverywhere();
  // Same name as someone else: probably the same person
  if (change.name) {
    const renamed = personById(person.id);
    const twin = renamed && people.people.find((p) => p.id !== renamed.id
      && p.name.trim().toLowerCase() === renamed.name.trim().toLowerCase());
    if (twin && confirm(`“${twin.name}” already exists. Are they the same person? OK merges them into one.`)) {
      await mergePeople([renamed.id], twin.id);
      return;
    }
  }
  refreshTakeCards();
}

/** Merge several people into one; the takes' speakers follow. */
async function mergePeople(fromIds, intoId) {
  for (const id of fromIds.filter((x) => x !== intoId)) {
    try {
      setPeople(await postJson(voiceUrl(`/speakers/${id}/merge`), { into: intoId }));
    } catch (err) {
      SMT.showError(err.message);
      break;
    }
    lastTakes.forEach((t) => (t.speakers || []).forEach((s) => { if (s.person === id) s.person = intoId; }));
    peopleUi.selected.delete(id);
  }
  selectTargetEverywhere();
  refreshTakeCards();
}

async function dismissSimilar(person, otherId) {
  try {
    setPeople(await postJson(voiceUrl(`/speakers/${person.id}/not-same`), { other: otherId }));
  } catch (err) {
    SMT.showError(err.message);
    return;
  }
  renderPeople();
}

/** Click a name to rename it in place (Enter saves, Esc cancels). */
function editableName(person, tag = 'strong') {
  const label = el(tag, { className: 'person-name', textContent: person.name, title: 'Click to rename' });
  label.tabIndex = 0;
  const startEditing = () => {
    const input = el('input', { type: 'text', className: 'name-input', value: person.name, maxLength: 40 });
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      input.replaceWith(label);
      if (save && value && value !== person.name) {
        label.textContent = value;
        updatePerson(person, { name: value });
      }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
    label.replaceWith(input);
    input.focus();
    input.select();
  };
  label.addEventListener('click', startEditing);
  label.addEventListener('keydown', (e) => { if (e.key === 'Enter') startEditing(); });
  return label;
}

function selectTargetEverywhere() {
  // "This is the voice": pick that person in every take that has them
  for (const take of lastTakes) {
    if (!take.speakers || !takeEdits[take.id]) continue;
    const ids = take.speakers.filter((s) => s.person && s.person === people.target).map((s) => s.id);
    if (people.target) takeEdits[take.id].speakers = new Set(ids);
  }
}

function refreshTakeCards() {
  renderPeople();
  Object.keys(renderedTakes).forEach((id) => { if (renderedTakes[id].startsWith('done')) delete renderedTakes[id]; });
  loadTakes();
}

function targetClips() {
  // Kept clips of the chosen person in every finished take (respecting edits)
  const out = [];
  for (const take of lastTakes) {
    if (take.state !== 'done' || !take.speakers) continue;
    const edits = takeEditsFor(take);
    const mine = new Set(take.speakers.filter((s) => s.person === people.target).map((s) => s.id));
    const clips = take.segments
      .map((segment, i) => ({ index: i, text: (edits.clips[i]?.text ?? segment.text).trim(), keep: edits.clips[i]?.keep !== false, speaker: segment.speaker }))
      .filter((c) => mine.has(c.speaker) && c.keep && c.text);
    if (clips.length) out.push({ take, clips });
  }
  return out;
}

function personCard(person) {
  const isTarget = person.id === people.target;
  const selected = peopleUi.selected.has(person.id);
  const card = el('div', { className: `person${isTarget ? ' selected' : ''}${selected ? ' ticked' : ''}` });

  const tick = el('input', { type: 'checkbox', checked: selected, title: 'Select to merge several people' });
  tick.addEventListener('change', () => {
    if (tick.checked) peopleUi.selected.add(person.id); else peopleUi.selected.delete(person.id);
    renderPeople();
  });
  const head = el('div', { className: 'speaker-head' }, tick, editableName(person));
  if (isTarget) head.append(el('span', { className: 'pill pill--ok', textContent: 'The voice' }));
  card.append(head);
  card.append(el('small', {
    className: 'hint',
    textContent: `${clock(person.seconds)} of speech · in ${person.takes.length} file${person.takes.length === 1 ? '' : 's'}`,
  }));
  if (person.ai) card.append(aiLine(person));

  const actions = el('div', { className: 'samples' });
  const sampleTake = person.sample && lastTakes.find((t) => t.id === person.sample.take && t.state === 'done');
  if (sampleTake) {
    const play = el('button', { type: 'button', className: 'btn btn--secondary', textContent: '▶', title: 'Hear them' });
    play.addEventListener('click', () => playClip(sampleTake, person.sample.index, play));
    actions.append(play);
  }
  const pick = el('button', {
    type: 'button', className: `btn ${isTarget ? 'btn--primary' : 'btn--secondary'}`,
    textContent: isTarget ? '✓ The voice I want' : 'This is the voice',
  });
  pick.addEventListener('click', () => updatePerson(person, { target: !isTarget }));
  actions.append(pick, focusButton([person.id], '◎ Focus'));
  const others = people.people.filter((p) => p.id !== person.id);
  if (others.length) {
    const merge = el('select', { title: 'The same person was found twice? Merge them.' },
      new Option('Same as…', ''), ...others.map((p) => new Option(p.name, p.id)));
    merge.addEventListener('change', async () => {
      const into = personById(merge.value);
      if (!into || !confirm(`Merge “${person.name}” into “${into.name}”? They'll be one person in every file.`)) {
        merge.value = '';
        return;
      }
      await mergePeople([person.id], into.id);
    });
    actions.append(merge);
  }
  card.append(actions);

  // Possibly the same person, found separately (e.g. split within a file)
  for (const hint of (person.similar || []).slice(0, 2)) {
    const other = personById(hint.id);
    if (!other) continue;
    const row = el('div', { className: 'maybe-same' });
    row.append(el('span', {
      textContent: hint.reason
        ? `Maybe the same as ${other.name}: ${hint.reason}`
        : `Maybe the same as ${other.name} (${Math.round(hint.score * 100)}%)`,
    }));
    const otherTake = other.sample && lastTakes.find((t) => t.id === other.sample.take && t.state === 'done');
    if (otherTake) {
      const play = el('button', { type: 'button', className: 'btn btn--ghost', textContent: '▶', title: `Hear ${other.name}` });
      play.addEventListener('click', () => playClip(otherTake, other.sample.index, play));
      row.append(play);
    }
    const yes = el('button', { type: 'button', className: 'btn btn--secondary', textContent: 'Merge' });
    yes.addEventListener('click', () => {
      // Keep the one with more speech (and its name), unless the other is named or chosen
      const keepOther = other.id === people.target || (other.seconds >= person.seconds && person.id !== people.target);
      const [from, into] = keepOther ? [person, other] : [other, person];
      mergePeople([from.id], into.id);
    });
    const no = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Not the same' });
    no.addEventListener('click', () => dismissSimilar(person, other.id));
    row.append(yes, no);
    card.append(row);
  }
  return card;
}

function aiLine(person) {
  const ai = person.ai;
  const row = el('div', { className: `ai-line ai-${ai.confidence}` });
  if (!ai.name) {
    row.append(el('span', { textContent: 'AI: not sure who this is' }));
  } else if (ai.name.toLowerCase() === 'mixed') {
    row.append(el('span', { textContent: 'AI: lines from several people' }));
  } else {
    row.append(el('span', { textContent: `AI: ${ai.name}${ai.actor ? ` (${ai.actor})` : ''} · ${ai.confidence}` }));
    if (ai.name !== person.name) {
      const use = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Use this name' });
      use.addEventListener('click', () => updatePerson(person, { name: ai.name.slice(0, 40) }));
      row.append(use);
    }
  }
  row.title = ai.reason || '';
  return row;
}

function identifyBar() {
  const info = people.identify || {};
  const bar = el('div', { className: 'identify-bar' });
  if (!info.connected) return bar;  // the Character clone banner says what's missing
  // Only the server knows if a run is going (a saved "running" is left over from a restart)
  const running = Boolean(info.running);
  const go = el('button', {
    type: 'button', className: 'btn btn--secondary', disabled: running,
    textContent: running ? 'Identifying…' : '🔎 Identify with AI',
    title: 'The AI reads the transcript of each file not done yet and says who speaks each line (Shift-click: all files again)',
  });
  go.addEventListener('click', async (e) => {
    go.disabled = true;
    try {
      setPeople(await postJson(voiceUrl('/speakers/identify'), { everyone: e.shiftKey }));
    } catch (err) {
      SMT.showError(err.message);
    }
    loadPeople();
  });
  bar.append(go);
  const ai = people.ai || {};
  const bits = [info.provider && `with ${info.provider}`, info.webSearch && 'web search on',
    ai.cast && `cast: ${ai.cast}`].filter(Boolean);
  let text = bits.join(' · ');
  if (running) text = `Asking the AI${ai.detail ? ` (${ai.detail})` : ''}… ${text}`;
  else if (ai.state === 'error') text = `Last try failed: ${ai.error}`;
  else if (ai.state === 'running') text = 'The last run was interrupted (restart). Press Identify to run it again.';
  else if (ai.at) text = `${text}${text ? ' · ' : ''}last run ${new Date(ai.at * 1000).toLocaleTimeString()}`;
  bar.append(el('span', { className: `hint${ai.state === 'error' && !running ? ' bad' : ''}`, textContent: text }));
  return bar;
}

/** Focus: the chosen cards, plus every card that may be the same character
 *  (same name or AI answer, or a voice match / AI suggestion either way). */
function focusVisible() {
  const focus = (people.focus || []).map(personById).filter(Boolean);
  if (!focus.length) return null;
  const keep = new Set(focus.map((p) => p.id));
  const characters = new Set(focus.map(characterOf).filter(Boolean));
  people.people.forEach((p) => {
    if (characters.has(characterOf(p))) keep.add(p.id);
  });
  people.people.forEach((p) => {
    (p.similar || []).forEach((hint) => {
      if (keep.has(p.id) && focus.some((f) => f.id === p.id)) keep.add(hint.id);
      if (focus.some((f) => f.id === hint.id)) keep.add(p.id);
    });
  });
  return keep;
}

async function setFocus(ids) {
  try {
    setPeople(await postJson(voiceUrl('/speakers/focus'), { ids }));
  } catch (err) {
    SMT.showError(err.message);
    return;
  }
  peopleUi.showHidden = false;
  renderPeople();
}

function focusButton(ids, label) {
  const focused = new Set(people.focus || []);
  const already = ids.every((id) => focused.has(id));
  const btn = el('button', {
    type: 'button', className: 'btn btn--ghost focus-btn',
    textContent: already ? '◎ Unfocus' : label,
    title: already ? 'Stop focusing on this' : 'Show only this character and cards that may be them; hide the rest',
  });
  btn.addEventListener('click', () => {
    const next = already ? [...focused].filter((id) => !ids.includes(id)) : [...new Set([...focused, ...ids])];
    setFocus(next);
  });
  return btn;
}

/** Who a card is: your name for it, else the AI's answer (medium or high confidence). */
function characterOf(person) {
  if (!/^Person \d+$/.test(person.name.trim())) return person.name.trim().toLowerCase().replace(/\s+/g, ' ');
  const ai = person.ai || {};
  const name = (ai.name || '').trim().toLowerCase().replace(/\s+/g, ' ');
  if (!name || ['mixed', 'unknown'].includes(name) || !['high', 'medium'].includes(ai.confidence)) return '';
  return name;
}

function sum(group) {
  return group.reduce((total, p) => total + p.seconds, 0);
}

function characterGroup(group) {
  const named = group.find((p) => !/^Person \d+$/.test(p.name.trim()));
  const title = named ? named.name : group[0].ai.name;
  const section = el('section', { className: 'character-group' });
  const head = el('div', { className: 'group-head' },
    el('strong', { textContent: title }),
    el('span', { className: 'hint', textContent: `${group.length} cards · ${clock(sum(group))} of speech` }));
  const into = [...group].sort((a, b) => (b.id === people.target) - (a.id === people.target) || b.seconds - a.seconds)[0];
  const mergeAll = el('button', {
    type: 'button', className: 'btn btn--secondary', textContent: `Merge all ${group.length}`,
    title: `One card for ${title}. Listen first: shouting or whispering may be worth keeping apart.`,
  });
  mergeAll.addEventListener('click', () => {
    if (confirm(`Merge all ${group.length} “${title}” cards into one? They'll be one person in every file.`)) {
      mergePeople(group.map((p) => p.id), into.id);
    }
  });
  head.append(mergeAll, focusButton(group.map((p) => p.id), `◎ Only ${title}`));
  section.append(head);
  const grid = el('div', { className: 'people' });
  group.forEach((person) => grid.append(personCard(person)));
  section.append(grid);
  return section;
}

function renderPeople() {
  const box = $('#people-panel');
  box.innerHTML = '';
  // Import file: the people in those files (Character clone files are reviewed per character instead)
  const importIds = new Set(lastTakes.filter((t) => !isCloneTake(t)).map((t) => t.id));
  const list = people.people.filter((p) => p.takes.some((id) => importIds.has(id)));
  show(box, list.length > 0 && recordMode !== 'clone');
  if (!list.length || recordMode === 'clone') return;

  const multi = lastTakes.filter((t) => t.speakers && !isCloneTake(t)).length > 1;
  const suggestions = list.filter((p) => (p.similar || []).length).length;
  const host = box;
  host.append(el('div', { className: 'people-head' },
    el('strong', { textContent: `People in your files (${list.length})` }),
    el('span', {
      className: 'hint',
      textContent: (multi ? 'Recognised across files by their voice. ' : '')
        + 'Click a name to rename. Mark the voice you want'
        + (suggestions ? '; “Maybe the same as” flags people who may have been split in two.' : '.'),
    })));

  // Tools for long lists: search, and merging several at once
  const tools = el('div', { className: 'people-tools' });
  if (list.length > 8) {
    const search = el('input', { type: 'search', placeholder: 'Find a person…', value: peopleUi.filter });
    search.addEventListener('input', () => {
      peopleUi.filter = search.value;
      const pos = search.selectionStart;
      renderPeople();
      const again = $('#people-panel input[type=search]');
      again.focus();
      again.setSelectionRange(pos, pos);
    });
    tools.append(search);
  }
  const ticked = [...peopleUi.selected].map(personById).filter(Boolean);
  if (ticked.length >= 2) {
    const into = [...ticked].sort((a, b) => (b.id === people.target) - (a.id === people.target) || b.seconds - a.seconds)[0];
    const mergeBtn = el('button', {
      type: 'button', className: 'btn btn--primary',
      textContent: `Merge ${ticked.length} selected into “${into.name}”`,
    });
    mergeBtn.addEventListener('click', () => {
      if (confirm(`Merge ${ticked.map((p) => p.name).join(', ')} into “${into.name}”? They'll be one person in every file.`)) {
        mergePeople(ticked.map((p) => p.id), into.id);
      }
    });
    const clear = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Clear selection' });
    clear.addEventListener('click', () => { peopleUi.selected.clear(); renderPeople(); });
    tools.append(mergeBtn, clear);
  } else if (list.length > 2) {
    tools.append(el('span', { className: 'hint', textContent: 'Tick two or more to merge them.' }));
  }
  if (tools.childNodes.length) host.append(tools);

  // Focus: only chosen characters and their possible matches
  const visible = focusVisible();
  const hiddenPeople = visible ? list.filter((p) => !visible.has(p.id)) : [];
  // The focused characters by name (a card's AI answer when you haven't named it)
  const focusTitles = [...new Set((people.focus || []).map(personById).filter(Boolean)
    .map((p) => (/^Person \d+$/.test(p.name.trim()) && p.ai?.name) || p.name))];
  if (visible) {
    const names = focusTitles;
    const banner = el('div', { className: 'focus-banner' },
      el('span', { textContent: `Showing only ${[...new Set(names)].join(', ')} and possible matches (${list.length - hiddenPeople.length} of ${list.length}).` }));
    const toggle = el('button', {
      type: 'button', className: 'btn btn--ghost',
      textContent: peopleUi.showHidden ? 'Hide the others' : `Show hidden (${hiddenPeople.length})`,
    });
    toggle.addEventListener('click', () => { peopleUi.showHidden = !peopleUi.showHidden; renderPeople(); });
    const clearFocus = el('button', { type: 'button', className: 'btn btn--secondary', textContent: 'Show everyone' });
    clearFocus.addEventListener('click', () => setFocus([]));
    banner.append(toggle, clearFocus);
    host.append(banner);
  }

  const needle = peopleUi.filter.trim().toLowerCase();
  const all = [...list]
    .filter((p) => !visible || visible.has(p.id))
    .filter((p) => !needle || p.name.toLowerCase().includes(needle)
      || (p.ai?.name || '').toLowerCase().includes(needle))
    .sort((a, b) => (b.id === people.target) - (a.id === people.target)
      || peopleUi.selected.has(b.id) - peopleUi.selected.has(a.id)
      || b.seconds - a.seconds);

  if (visible) {
    // Chosen characters: confirmed cards first, then the ones that may be them
    const focusIds = new Set(people.focus || []);
    const focusChars = new Set((people.focus || []).map(personById).filter(Boolean).map(characterOf).filter(Boolean));
    // Confirmed: named by you, or the AI is sure; a focused card with no name yet counts too
    const isConfirmed = (p) => (focusChars.has(characterOf(p))
      ? !/^Person \d+$/.test(p.name.trim()) || p.ai?.confidence === 'high'
      : focusIds.has(p.id));
    const confirmed = all.filter(isConfirmed);
    const possible = all.filter((p) => !isConfirmed(p));
    const section = (title, hint, cards, mergeable) => {
      if (!cards.length) return;
      const head = el('div', { className: 'group-head' },
        el('strong', { textContent: title }),
        el('span', { className: 'hint', textContent: hint }));
      if (mergeable && cards.length > 1) {
        const into = [...cards].sort((a, b) => (b.id === people.target) - (a.id === people.target) || b.seconds - a.seconds)[0];
        const mergeAll = el('button', { type: 'button', className: 'btn btn--secondary', textContent: `Merge all ${cards.length}` });
        mergeAll.addEventListener('click', () => {
          if (confirm(`Merge the ${cards.length} confirmed cards into “${into.name}”? Listen first: shouting or whispering may be worth keeping apart.`)) {
            mergePeople(cards.map((c) => c.id), into.id);
          }
        });
        head.append(mergeAll);
      }
      const grid = el('div', { className: 'people' });
      cards.forEach((person) => grid.append(personCard(person)));
      host.append(el('section', { className: 'character-group' }, head, grid));
    };
    section(`Confirmed (${confirmed.length})`, `${clock(sum(confirmed))} of speech · named by you, or the AI is sure`, confirmed, true);
    section(`Possible matches (${possible.length})`,
      'The AI is less sure, or the voice sounds alike. ▶ to check, then Merge into a confirmed card or Not the same.', possible, false);
  } else {
    // Cards of the same character side by side (same name, or the same AI answer):
    // often one person split by shouting, whispering or a cold. Merging stays your call.
    const groups = new Map();
    all.forEach((p) => {
      const key = characterOf(p) || `#${p.id}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(p);
    });
    const split = [...groups.values()].filter((g) => g.length > 1)
      .sort((a, b) => sum(b) - sum(a));
    split.forEach((group) => host.append(characterGroup(group)));

    const sorted = all.filter((p) => groups.get(characterOf(p) || `#${p.id}`).length === 1);
    const shown = needle || peopleUi.showAll ? sorted : sorted.slice(0, PEOPLE_SHOWN);
    const grid = el('div', { className: 'people' });
    shown.forEach((person) => grid.append(personCard(person)));
    if (shown.length) {
      if (split.length) host.append(el('strong', { className: 'group-title', textContent: 'Everyone else' }));
      host.append(grid);
    }
    if (shown.length < sorted.length) {
      const more = el('button', {
        type: 'button', className: 'btn btn--ghost',
        textContent: `Show all ${sorted.length} people (${sorted.length - shown.length} more with less speech)`,
      });
      more.addEventListener('click', () => { peopleUi.showAll = true; renderPeople(); });
      host.append(more);
    } else if (peopleUi.showAll && sorted.length > PEOPLE_SHOWN && !needle) {
      const less = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Show fewer' });
      less.addEventListener('click', () => { peopleUi.showAll = false; renderPeople(); });
      host.append(less);
    }

  }

  if (visible && peopleUi.showHidden && hiddenPeople.length) {
    host.append(el('strong', { className: 'group-title', textContent: `Hidden (${hiddenPeople.length}): not ${focusTitles.join(' or ')}, as far as names and voices tell` }));
    const hiddenGrid = el('div', { className: 'people hidden-people' });
    [...hiddenPeople].sort((a, b) => b.seconds - a.seconds)
      .filter((p) => !needle || p.name.toLowerCase().includes(needle) || (p.ai?.name || '').toLowerCase().includes(needle))
      .forEach((person) => hiddenGrid.append(personCard(person)));
    host.append(hiddenGrid);
  }

  const target = personById(people.target);
  if (target) {
    const batches = targetClips();
    const total = batches.reduce((sum, b) => sum + b.clips.length, 0);
    if (total) {
      const saveAll = el('button', {
        type: 'button', className: 'btn btn--primary',
        textContent: `✓ Save ${target.name}'s ${total} clips from ${batches.length} file${batches.length === 1 ? '' : 's'} to the dataset`,
      });
      saveAll.addEventListener('click', async () => {
        if (!confirm(`Save ${total} clips of ${target.name} from ${batches.length} files? Unticked clips stay out, and the other people's clips are discarded with those files' reviews.`)) return;
        saveAll.disabled = true;
        let saved = 0;
        for (const [i, batch] of batches.entries()) {
          saveAll.textContent = `Saving file ${i + 1} of ${batches.length}…`;
          try {
            const result = await postJson(voiceUrl(`/freeform/${batch.take.id}/save`), {
              clips: batch.clips.map(({ index, text }) => ({ index, text })),
            });
            saved += result.saved;
            delete takeEdits[batch.take.id];
            updateRecorded(result.recorded);
          } catch (err) {
            SMT.showError(`${batch.take.name || batch.take.id}: ${err.message}`);
          }
        }
        $('#free-upload-status').textContent = `Added ${saved} clips of ${target.name} to the dataset.`;
        loadTakes();
      });
      host.append(el('div', { className: 'take-foot' },
        el('span', { className: 'hint', textContent: 'Review the clips in each file below first; your edits and unticks are kept.' }), saveAll));
    }
  }
}

// ---------------------------------------------------------------------------
// Character clone: files -> speakers -> AI reads who says each line -> pick a character -> review

const cloneUi = { filesOpen: true, showChars: false };

/** Where a Character clone file is in the process: [text, style]. */
function cloneStage(take) {
  const ai = people.ai || {};
  if (take.state === 'running') {
    return /waiting for the other files/i.test(take.detail || '')
      ? ['Queued', '']
      : [`Preparing: ${take.detail || 'working…'}`, 'pill--warn'];
  }
  if (take.state === 'choose_track') return ['Needs an audio track choice (below)', 'pill--warn'];
  if (take.state === 'error') return ['Failed (below)', 'pill--bad'];
  if (people.identify?.running && ai.take === take.id) {
    return [`AI reading${ai.parts > 1 ? ` · part ${ai.part} of ${ai.parts}` : ''}`, 'pill--warn'];
  }
  if (!take.attributed) return ['Waiting for the AI', ''];
  // Clips saved from it (of the character picked, or any): reviewed
  if (cloneFileStats(take, charUi.name).added > 0) return ['✓ Reviewed', ''];
  return ['Ready for review', 'pill--ok'];
}

/** One file's numbers for a character (or, without one, just what's in the dataset). */
function cloneFileStats(take, name) {
  const segments = take.segments || [];
  const saved = segments.filter((s) => s.saved);
  if (!name) return { added: saved.length };
  const owners = take.groupCharacters || {};
  const mine = segments.filter((s) => s.character === name
    || (s.character == null && 'character' in s && owners[String(s.speaker)] === name));
  const added = mine.filter((s) => s.saved);
  const reviewed = added.length > 0;  // same rule as the review: saved here = file reviewed
  return {
    confirmed: mine.filter((s) => s.confirmed && s.character === name).length,
    possible: mine.filter((s) => !(s.confirmed && s.character === name)).length,
    added: added.length,
    edited: added.filter((s) => s.savedText && s.savedText !== s.text).length,
    leftOut: reviewed ? mine.length - added.length : 0,
    toReview: reviewed ? 0 : mine.length,
  };
}

/** Every Character clone file: its stage, and what the review did with it. */
function cloneFilesTable(takes) {
  const name = charUi.name;
  const details = el('details', { className: 'clone-files' });
  details.open = cloneUi.filesOpen;
  details.addEventListener('toggle', () => { cloneUi.filesOpen = details.open; });
  details.append(el('summary', { textContent: `Files (${takes.length})` }));

  const cols = name
    ? ['File', 'Stage', 'Lines', `${name}: ✓ / ?`, 'Added', 'Edited', 'Left out', 'To review', '']
    : ['File', 'Stage', 'Lines', 'Added to dataset', ''];
  const table = el('table', { className: 'clone-table' });
  table.append(el('thead', {}, el('tr', {}, ...cols.map((c, i) => el('th', { textContent: c, className: i > 1 ? 'num' : '' })))));
  const body = el('tbody');
  const totals = { lines: 0, confirmed: 0, possible: 0, added: 0, edited: 0, leftOut: 0, toReview: 0 };
  // Oldest first, like the character review (take ids are upload timestamps)
  const sorted = [...takes].sort((a, b) => (a.created || 0) - (b.created || 0) || a.id.localeCompare(b.id));
  sorted.forEach((take) => {
    const [stage, style] = cloneStage(take);
    const lines = take.state === 'done' ? (take.segments || []).length : null;
    const s = take.state === 'done' ? cloneFileStats(take, name) : null;
    if (lines) totals.lines += lines;
    if (s) Object.keys(s).forEach((k) => { totals[k] += s[k]; });
    const discard = el('button', { type: 'button', className: 'btn btn--ghost', textContent: '✕', title: 'Discard this file' });
    discard.addEventListener('click', async () => {
      if (!confirm(`Discard ${take.name || 'this file'}? Clips already in the dataset stay there.`)) return;
      try {
        await api(voiceUrl(`/freeform/${take.id}`), { method: 'DELETE' });
      } catch (err) {
        SMT.showError(err.message);
      }
      loadTakes();
    });
    const num = (v) => el('td', { className: 'num', textContent: v == null ? '' : String(v) });
    const cells = [
      el('td', { className: 'file', textContent: take.name || take.id, title: take.name || take.id }),
      el('td', { className: 'file-stage' }, el('span', { className: `pill ${style}`, textContent: stage })),
      num(lines),
    ];
    if (name) {
      cells.push(num(s ? `${s.confirmed} / ${s.possible}` : null), num(s?.added), num(s?.edited), num(s?.leftOut), num(s?.toReview));
    } else {
      cells.push(num(s?.added));
    }
    cells.push(el('td', {}, take.state === 'running' ? '' : discard));
    body.append(el('tr', {}, ...cells));
  });
  table.append(body);
  const foot = name
    ? ['Total', '', totals.lines, `${totals.confirmed} / ${totals.possible}`, totals.added, totals.edited, totals.leftOut, totals.toReview, '']
    : ['Total', '', totals.lines, totals.added, ''];
  table.append(el('tfoot', {}, el('tr', {}, ...foot.map((v, i) => el('td', { textContent: String(v), className: i > 1 ? 'num' : '' })))));
  details.append(el('div', { className: 'table-wrap' }, table));
  return details;
}

/** Character clone files the AI still has to read (with an AI connected). */
function awaitingAi(takes) {
  return Boolean(people.identify?.connected)
    && takes.some((t) => isCloneTake(t) && t.state === 'done' && !t.attributed);
}

/** A Character clone file (or one the AI already read before that tab existed). */
function isCloneTake(take) {
  return Boolean(take.clone || take.attributed);
}

/** Each tab lists its own files. */
function applyTakeFilter() {
  const clone = recordMode === 'clone';
  const byId = new Map(lastTakes.map((t) => [t.id, t]));
  document.querySelectorAll('#free-takes .take').forEach((node) => {
    const take = byId.get(node.dataset.id);
    // Character clone: the Files table shows every file; below, only those needing you (failed, track choice)
    const needed = !clone || ['error', 'choose_track'].includes(take?.state);
    show(node, Boolean(take) && isCloneTake(take) === clone && needed);
  });
}

function renderClone() {
  if (recordMode !== 'clone') return;
  const info = people.identify || {};
  const banner = $('#clone-ai');
  show(banner, !info.connected);
  banner.innerHTML = '';
  if (!info.connected) {
    banner.append('Character clone needs an AI to read the dialogue. ',
      el('a', { href: '../#settings', target: '_top', textContent: 'Add an AI connection in Settings' }), '.');
  }

  // Progress over all Character clone files
  const takes = lastTakes.filter(isCloneTake);
  const progress = $('#clone-progress');
  progress.innerHTML = '';
  show(progress, takes.length > 0);
  if (takes.length) {
    const running = takes.filter((t) => t.state === 'running');
    const count = (f) => takes.filter(f).length;
    const bits = [
      `${takes.length} file${takes.length === 1 ? '' : 's'}`,
      running.length && `${running.length} being prepared (${running[0].name || 'file'}: ${running[0].detail || 'working…'})`,
      count((t) => t.state === 'choose_track') && `${count((t) => t.state === 'choose_track')} need an audio track choice (below)`,
      count((t) => t.state === 'error') && `${count((t) => t.state === 'error')} failed (below)`,
      count((t) => t.state === 'done' && !t.attributed) && `${count((t) => t.state === 'done' && !t.attributed)} waiting for the AI`,
      count((t) => t.attributed && !cloneFileStats(t, charUi.name).added)
        && `${count((t) => t.attributed && !cloneFileStats(t, charUi.name).added)} ready for review`,
      count((t) => t.attributed && cloneFileStats(t, charUi.name).added)
        && `${count((t) => t.attributed && cloneFileStats(t, charUi.name).added)} reviewed`,
    ].filter(Boolean);
    progress.append(el('span', { textContent: bits.join(' · ') }));
    progress.append(identifyBar());
    progress.append(cloneFilesTable(takes));
  }

  const box = $('#clone-characters');
  box.innerHTML = '';
  const chars = peopleUi.characters || [];
  const current = chars.find((c) => c.name === charUi.name);
  if (current && current.saved > 0 && !cloneUi.showChars) {
    // Working on one character (some clips saved): the others fold away
    const change = el('button', { type: 'button', className: 'btn btn--secondary', textContent: 'Change character' });
    change.addEventListener('click', () => { cloneUi.showChars = true; renderClone(); });
    box.append(el('div', { className: 'character-current' },
      el('span', { className: 'hint', textContent: 'Character:' }),
      el('strong', { textContent: current.name }),
      el('span', { className: 'hint', textContent: `${current.confirmed} ✓ · ${current.possible} ? · ${current.saved} saved` }),
      change));
  } else if (chars.length) {
    box.append(characterList(chars));
    if (current && current.saved > 0) {
      const hide = el('button', { type: 'button', className: 'btn btn--ghost', textContent: `Keep ${current.name}` });
      hide.addEventListener('click', () => { cloneUi.showChars = false; renderClone(); });
      box.querySelector('.character-list').append(hide);
    }
  }
  // Come back to the character picked last time
  if (!charUi.name && chars.length) {
    let saved = '';
    try { saved = localStorage.getItem(`clone:${voice.name}`) || ''; } catch (e) { /* ignore */ }
    if (chars.some((c) => c.name === saved)) openCharacter(saved);
  }
}

/** Chips for every character the AI found; click one to review their clips. */
function characterList(chars) {
  const box = el('div', { className: 'character-list' },
    el('span', { className: 'hint', textContent: 'Characters (who says what, from the AI reading every line):' }));
  chars.forEach((c) => {
    const chip = el('button', {
      type: 'button', className: `chip${charUi.name === c.name ? ' chip--on' : ''}`,
      textContent: `${c.name} · ${c.confirmed} ✓${c.possible ? ` · ${c.possible} ?` : ''}`,
      title: `${c.confirmed} confirmed clips (${clock(c.seconds)}), ${c.possible} possible, in ${c.files} file${c.files === 1 ? '' : 's'}`
        + (c.saved ? ` · ${c.saved} already saved` : ''),
    });
    chip.addEventListener('click', () => {
      cloneUi.showChars = false;  // picked: the list folds away again (once there are saves)
      if (charUi.name === c.name) closeCharacter(); else openCharacter(c.name);
    });
    box.append(chip);
  });
  return box;
}

const charUi = { name: null, data: null, keep: new Map(), text: new Map(), open: new Set() };

async function openCharacter(name, keepChoices = false) {
  const panel = $('#character-panel');
  if (!keepChoices) {
    Object.assign(charUi, { name, data: null, keep: new Map(), text: new Map(), open: new Set(), sig: '' });
    panel.innerHTML = '';
    panel.append(el('p', { className: 'hint', textContent: `Loading ${name}'s clips…` }));
    try { localStorage.setItem(`clone:${voice.name}`, name); } catch (e) { /* ignore */ }
  }
  show(panel, true);
  renderClone();
  try {
    charUi.data = await api(voiceUrl(`/characters/clips?character=${encodeURIComponent(name)}`));
  } catch (err) {
    SMT.showError(err.message);
    closeCharacter();
    return;
  }
  if (charUi.name !== name) return;
  renderCharacter();
  if (!keepChoices) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function closeCharacter() {
  charUi.name = null;
  try { localStorage.removeItem(`clone:${voice.name}`); } catch (e) { /* ignore */ }
  show($('#character-panel'), false);
  renderClone();
}

const clipKey = (clip) => `${clip.take}:${clip.index}`;

function renderCharacter() {
  const panel = $('#character-panel');
  const { name, confirmed, possible } = charUi.data;
  // Rebuilding the panel (e.g. after Save) keeps you where you were in each file's list
  const scrolls = new Map([...panel.querySelectorAll('details.char-file')]
    .map((d) => [d.dataset.id, d.querySelector('.clips')?.scrollTop || 0]));
  // Which episodes were on screen and open: a refresh keeps them as they were
  const shown = new Set([...panel.querySelectorAll('details.char-file')].map((d) => d.dataset.id));
  const wasOpen = new Set([...panel.querySelectorAll('details.char-file')].filter((d) => d.open).map((d) => d.dataset.id));
  const pageY = window.scrollY;
  panel.innerHTML = '';
  const files = new Set([...confirmed, ...possible].map((c) => c.take)).size;
  const close = el('button', { type: 'button', className: 'btn btn--ghost', textContent: '✕ Close' });
  close.addEventListener('click', closeCharacter);
  panel.append(el('div', { className: 'group-head' },
    el('h3', { textContent: name }),
    el('span', { className: 'hint', textContent: `${confirmed.length} confirmed · ${possible.length} possible · in ${files} file${files === 1 ? '' : 's'}` }),
    close));
  panel.append(el('p', {
    className: 'hint',
    textContent: 'Confirmed: the AI is sure it\'s them and the voice agrees. Possible: check these (▶), tick the ones that really are them. '
      + 'Untick lines with music, shouting, whispering or other voices; fix wrong words.',
  }));

  const save = el('button', { type: 'button', className: 'btn btn--primary' });
  // Saved clips whose words were changed here: corrected in the dataset with the same Save
  const corrections = () => [...confirmed, ...possible].filter((c) => c.saved
    && charUi.text.has(clipKey(c)) && charUi.text.get(clipKey(c)).trim() && charUi.text.get(clipKey(c)).trim() !== c.text);
  const updateSave = () => {
    const n = [...confirmed, ...possible].filter((c) => !c.saved && ticked(c)).length;
    const fixes = corrections().length;
    save.textContent = `✓ Save ${n} ticked clip${n === 1 ? '' : 's'}`
      + (fixes ? ` and ${fixes} correction${fixes === 1 ? '' : 's'}` : '') + ` of ${name} to the dataset`;
    save.disabled = n === 0 && fixes === 0;
  };
  // Default: confirmed ticked, possible not. A file with saved clips has been reviewed,
  // so what wasn't saved there was left out on purpose: unticked.
  const reviewed = new Set([...confirmed, ...possible].filter((c) => c.saved).map((c) => c.take));
  const ticked = (clip) => charUi.keep.get(clipKey(clip)) ?? (!clip.possible && !reviewed.has(clip.take));

  const section = (title, clips, possibleSection) => {
    if (!clips.length) return;
    clips.forEach((c) => { c.possible = possibleSection; });
    const head = el('div', { className: 'group-head' }, el('strong', { textContent: `${title} (${clips.length})` }));
    const all = el('button', { type: 'button', className: 'btn btn--ghost', textContent: possibleSection ? 'Tick all' : 'Untick all' });
    all.addEventListener('click', () => {
      clips.forEach((c) => { if (!c.saved) charUi.keep.set(clipKey(c), possibleSection); });
      renderCharacter();
    });
    head.append(all);
    const box = el('section', { className: 'character-group' }, head);
    const byFile = new Map();
    clips.forEach((c) => { if (!byFile.has(c.take)) byFile.set(c.take, []); byFile.get(c.take).push(c); });
    [...byFile.values()].forEach((fileClips) => {
      const id = `${possibleSection ? 'p' : 'c'}:${fileClips[0].take}`;
      const details = el('details', { className: 'char-file' });
      details.dataset.id = id;
      // Open: what was open before a refresh; new episodes not reviewed yet (nothing of this
      // character saved from them) open too, so the ones just added are ready to go
      const fresh = !possibleSection && !reviewed.has(fileClips[0].take);
      details.open = shown.has(id) ? wasOpen.has(id) : fresh;
      if (details.open) charUi.open.add(id);
      const label = el('span', { className: 'char-file-name', textContent: fileSummary(fileClips, ticked) });
      const refresh = () => { label.textContent = fileSummary(fileClips, ticked); updateSave(); };
      // Tick or untick a whole file, open or not. Untick all also takes its saved clips
      // out of the dataset (like unticking them one by one); Tick all leaves them saved.
      const tickFile = (on) => async (e) => {
        e.preventDefault();  // a button in the summary would otherwise open/close the file
        e.stopPropagation();
        const savedHere = on ? [] : fileClips.filter((c) => c.saved);
        if (savedHere.length) {
          try {
            const result = await postJson(voiceUrl('/clips/unsave'), {
              clips: savedHere.map((c) => ({ take: c.take, index: c.index })),
            });
            updateRecorded(result.recorded);
          } catch (err) {
            SMT.showError(err.message);
            return;
          }
          savedHere.forEach((c) => { c.saved = false; });
          loadTakes();  // the Files table
        }
        fileClips.forEach((c) => { if (!c.saved) charUi.keep.set(clipKey(c), on); });
        const list = details.querySelector('.clips');
        if (list) {
          // Redraw this file's rows in place (no jump in the list)
          const top = list.scrollTop;
          list.replaceChildren(...fileClips.map((clip) => characterClipRow(clip, ticked, refresh)));
          list.scrollTop = top;
        }
        refresh();
      };
      const tickAll = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Tick all' });
      tickAll.addEventListener('click', tickFile(true));
      const untickAll = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Untick all',
        title: 'Untick every clip of this file, and take its saved clips out of the dataset' });
      untickAll.addEventListener('click', tickFile(false));
      const tools = el('span', { className: 'char-file-tools' }, tickAll, untickAll);
      details.append(el('summary', {}, label, tools));
      const fill = () => {
        if (details.dataset.filled) return;
        details.dataset.filled = '1';
        const list = el('div', { className: 'clips' });
        fileClips.forEach((clip) => list.append(characterClipRow(clip, ticked, refresh)));
        details.append(list);
        list.scrollTop = scrolls.get(id) || 0;
      };
      details.addEventListener('toggle', () => {
        if (details.open) { charUi.open.add(id); fill(); } else charUi.open.delete(id);
      });
      if (details.open) fill();
      box.append(details);
    });
    panel.append(box);
  };
  section('Confirmed', confirmed, false);
  section('Possible', possible, true);
  if (!confirmed.length && !possible.length) {
    panel.append(el('p', { className: 'hint', textContent: 'No clips of this character.' }));
  }

  save.addEventListener('click', async () => {
    const clips = [...confirmed, ...possible].filter((c) => !c.saved && ticked(c))
      .map((c) => ({ take: c.take, index: c.index, text: (charUi.text.get(clipKey(c)) ?? c.text).trim() }))
      .filter((c) => c.text);
    const fixes = corrections().map((c) => ({ take: c.take, index: c.index, text: charUi.text.get(clipKey(c)).trim() }));
    save.disabled = true;
    save.textContent = 'Saving…';
    const done = [];
    try {
      if (clips.length) {
        const result = await postJson(voiceUrl('/clips/save'), { clips });
        updateRecorded(result.recorded);
        done.push(`added ${result.saved} clips`);
      }
      if (fixes.length) {
        const result = await postJson(voiceUrl('/clips/text'), { clips: fixes });
        fixes.forEach((c) => charUi.text.delete(`${c.take}:${c.index}`));
        done.push(`corrected ${result.updated} clip${result.updated === 1 ? '' : 's'}`);
      }
      $('#free-upload-status').textContent = `${name}: ${done.join(' and ')} in the dataset.`;
    } catch (err) {
      SMT.showError(err.message);
    }
    openCharacter(name, true);  // shows them as saved; your ticks and edits stay
    loadTakes();  // the Files table's Added / Edited / Left out
  });
  panel.append(el('div', { className: 'take-foot' }, save));
  updateSave();
  // Scroll positions only take once the lists are on the page
  panel.querySelectorAll('details.char-file').forEach((d) => {
    const list = d.querySelector('.clips');
    if (list && scrolls.has(d.dataset.id)) list.scrollTop = scrolls.get(d.dataset.id);
  });
  if (scrolls.size) window.scrollTo(0, pageY);
}

/** "S01E03.mkv · 12 clips · 5 saved · 4 ticked" */
function fileSummary(clips, ticked) {
  const saved = clips.filter((c) => c.saved).length;
  const n = clips.filter((c) => !c.saved && ticked(c)).length;
  return [clips[0].file, `${clips.length} clip${clips.length === 1 ? '' : 's'}`, saved && `${saved} saved`,
    `${n} ticked`].filter(Boolean).join(' · ');
}

function characterClipRow(clip, ticked, changed) {
  const key = clipKey(clip);
  // Saved clips stay in the list, coloured, so you can see what's already in the dataset
  const row = el('div', { className: `clip${clip.saved ? ' clip--saved' : ticked(clip) ? '' : ' skipped'}` });
  const keep = el('input', { type: 'checkbox', checked: clip.saved || ticked(clip),
    title: clip.saved ? 'In the dataset: untick to take it out' : 'Save this clip' });
  keep.addEventListener('change', async () => {
    if (clip.saved) {
      // Out of the dataset right away, no questions; only this row changes (no jump in the list)
      keep.disabled = true;
      try {
        const result = await postJson(voiceUrl('/clips/unsave'), { clips: [{ take: clip.take, index: clip.index }] });
        updateRecorded(result.recorded);
      } catch (err) {
        SMT.showError(err.message);
        keep.checked = true;
        keep.disabled = false;
        return;
      }
      clip.saved = false;
      charUi.keep.set(key, false);
      outer.replaceWith(characterClipRow(clip, ticked, changed));
      changed();
      loadTakes();  // the Files table
      return;
    }
    charUi.keep.set(key, keep.checked);
    row.classList.toggle('skipped', !keep.checked);
    changed();
  });
  const play = el('button', { type: 'button', className: 'btn btn--secondary', textContent: '▶' });
  play.addEventListener('click', () => playClip({ id: clip.take, denoise: clip.denoise }, clip.index, play));
  const text = el('input', { type: 'text', value: charUi.text.get(key) ?? clip.text });
  const status = el('span', { className: 'dur', textContent: clip.saved ? '✓ saved' : `${(clip.end - clip.start).toFixed(1)}s`,
    title: `at ${clock(clip.start)} in ${clip.file}` });
  const markEdited = () => {
    // A saved clip with new words: corrected in the dataset with the next Save
    const edited = clip.saved && text.value.trim() !== clip.text;
    row.classList.toggle('clip--edited', edited);
    if (clip.saved) status.textContent = edited ? 'edited' : '✓ saved';
  };
  if (clip.saved) text.title = 'In the dataset: fix the words, then Save to update it';
  text.addEventListener('input', () => {
    charUi.text.set(key, text.value);
    markEdited();
    changed();
  });
  markEdited();
  row.append(keep, play, text, status);
  const outer = clip.reason
    ? el('div', { className: 'clip-with-reason' }, row, el('small', { className: 'hint', textContent: clip.reason }))
    : row;
  return outer;
}

async function loadTakes() {
  clearTimeout(takesTimer);
  if (!voice) return;
  const name = voice.name;
  let takes = [];
  try {
    ({ takes } = await api(voiceUrl('/freeform')));
  } catch (err) {
    return;
  }
  if (!voice || voice.name !== name) return;
  lastTakes = takes;
  await loadPeople();  // also tells whether an AI is connected (Character clone)

  const box = $('#free-takes');
  const ids = new Set(takes.map((t) => t.id));
  box.querySelectorAll('.take').forEach((node) => {
    if (!ids.has(node.dataset.id)) { node.remove(); delete renderedTakes[node.dataset.id]; }
  });
  takes.slice().reverse().forEach((take) => {
    const existing = box.querySelector(`.take[data-id="${take.id}"]`);
    if (existing && renderedTakes[take.id] === takeKey(take) && take.state !== 'running') return;
    const node = renderTake(take);
    if (existing) existing.replaceWith(node); else box.append(node);
    renderedTakes[take.id] = takeKey(take);
  });
  applyTakeFilter();
  renderClone();
  if (recordMode === 'prompts' && takes.some((t) => ['done', 'choose_track'].includes(t.state) && !isCloneTake(t))) {
    // Something is waiting for review: show it where it came from
    setMode(takes.some((t) => t.tracks && t.source && !/^source\.(webm|ogg|m4a)$/.test(t.source)) ? 'file' : 'free');
  }
  if (takes.some((t) => t.state === 'running')) {
    takesTimer = setTimeout(loadTakes, 2000);
  } else if (awaitingAi(takes)) {
    // Prepared, and the AI picks it up next on the server: keep watching until it has been read
    // (more slowly after a failed run: a new file or 🔎 Identify with AI starts it again)
    const failed = people.ai?.state === 'error' && !people.identify?.running;
    takesTimer = setTimeout(loadTakes, failed ? 15000 : 4000);
  }
  updatePending(takes);
}

// Clips from recorded/imported takes only join the dataset once saved:
// say so next to the counter and in the Train step.
let pendingClips = 0;
let pendingTakes = 0;
/** Multi-speaker files are reviewed per character (AI identification is on). */
const takeKey = (take) => `${take.state}${take.attributed ? ':ai' : ''}`;

function updatePending(takes) {
  // Character clone files are reviewed per character: never "waiting" here
  const waiting = takes.filter((t) => ['done', 'choose_track', 'running'].includes(t.state)
    && !(t.state === 'done' && isCloneTake(t)));
  pendingTakes = waiting.length;
  pendingClips = waiting.reduce((sum, t) => sum + (t.state === 'done' ? t.segments.length : 0), 0);
  const what = pendingClips
    ? `${pendingClips} clip${pendingClips === 1 ? '' : 's'} waiting for review`
    : `${pendingTakes} take${pendingTakes === 1 ? '' : 's'} still being prepared or waiting for a choice`;
  const note = $('#pending-note');
  note.textContent = pendingTakes ? ` · ${what}` : '';
  show(note, pendingTakes > 0);
  $('#train-pending-text').textContent = pendingTakes
    ? `${what[0].toUpperCase()}${what.slice(1)} in 2. Dataset. They're not in the dataset until you press ✓ Save in their review.`
    : '';
  show($('#train-pending'), pendingTakes > 0);
}

function goToReview() {
  if (VIEW !== 'dataset') {
    // The review is in the Build Dataset tab
    rememberVoice(voice?.name);
    window.top.location.hash = 'dataset';
    return;
  }
  if (recordMode === 'prompts') setMode('free');
  const first = document.querySelector('#free-takes .take');
  (first || $('#record-card')).scrollIntoView({ behavior: 'smooth', block: 'start' });
}
$('#train-pending-btn').addEventListener('click', goToReview);

function takeTitle(take) {
  const m = take.id.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
  const when = m ? `${m[2]}/${m[3]} ${m[4]}:${m[5]}` : take.id;
  const imported = take.name && !/^take\.(webm|ogg|m4a)$/.test(take.name);
  const parts = [imported ? take.name : `Take ${when}`];
  if (take.duration) parts.push(clock(take.duration));
  if (take.tracks && take.tracks.length > 1 && take.state !== 'choose_track') {
    parts.push(trackLabel(take.tracks[take.track]));
  }
  if (take.dialogue && take.state !== 'choose_track') parts.push('dialogue channel');
  if (take.speakers) parts.push(`${take.speakers.length} speaker${take.speakers.length === 1 ? '' : 's'}`);
  return parts.join(' · ');
}

const LANGUAGE_NAMES = {
  eng: 'English', spa: 'Spanish', fra: 'French', fre: 'French', deu: 'German', ger: 'German',
  ita: 'Italian', por: 'Portuguese', nld: 'Dutch', dut: 'Dutch', rus: 'Russian', pol: 'Polish',
  jpn: 'Japanese', kor: 'Korean', zho: 'Chinese', chi: 'Chinese', hin: 'Hindi', ara: 'Arabic',
  swe: 'Swedish', dan: 'Danish', nor: 'Norwegian', fin: 'Finnish', tur: 'Turkish', ukr: 'Ukrainian',
};

function trackLabel(track) {
  const parts = [LANGUAGE_NAMES[track.language] || (track.language ? track.language.toUpperCase() : `Track ${track.index + 1}`)];
  if (track.title) parts.push(track.title);
  parts.push(track.layout || `${track.channels} ch`, track.codec.toUpperCase());
  return parts.join(' · ');
}

const speakerName = (id) => `Speaker ${String.fromCharCode(65 + (id % 26))}${id >= 26 ? Math.floor(id / 26) + 1 : ''}`;
const SUGGEST_SIMILARITY = 0.45;

function takeEditsFor(take) {
  let edits = takeEdits[take.id];
  if (!edits) {
    edits = { clips: {}, speakers: new Set() };
    const target = (take.speakers || []).filter((s) => s.person && s.person === people.target);
    if (target.length) {
      // "The voice I want", recognised in this file
      target.forEach((s) => edits.speakers.add(s.id));
    } else {
      // Otherwise the speaker who sounds like this voice's existing recordings
      const best = (take.speakers || []).filter((s) => s.similarity !== null)
        .sort((a, b) => b.similarity - a.similarity)[0];
      if (best && best.similarity >= SUGGEST_SIMILARITY) edits.speakers.add(best.id);
    }
    takeEdits[take.id] = edits;
  }
  return edits;
}

function renderTake(take) {
  const box = el('div', { className: 'take' });
  box.dataset.id = take.id;
  const head = el('div', { className: 'take-head' }, el('strong', { textContent: takeTitle(take) }));
  box.append(head);

  const discard = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Discard' });
  discard.addEventListener('click', async () => {
    if (take.state === 'done' && !confirm('Discard this take and all its clips?')) return;
    try {
      await api(voiceUrl(`/freeform/${take.id}`), { method: 'DELETE' });
    } catch (err) {
      SMT.showError(err.message);
    }
    loadTakes();
  });

  if (take.state === 'running') {
    head.append(el('span', { className: 'pill pill--warn pill--live', textContent: take.detail || 'Working…' }));
    return box;
  }
  if (take.state === 'error') {
    const retry = el('button', { type: 'button', className: 'btn btn--secondary', textContent: '↻ Try again' });
    retry.title = 'Process it again from the file already uploaded';
    retry.addEventListener('click', async () => {
      retry.disabled = true;
      try {
        await postJson(voiceUrl(`/freeform/${take.id}/retry`), {});
      } catch (err) {
        SMT.showError(err.message);
        retry.disabled = false;
        return;
      }
      delete renderedTakes[take.id];
      loadTakes();
    });
    head.append(retry, discard);
    box.append(el('div', { className: 'banner banner--error', textContent: take.error || 'Transcription failed' }));
    return box;
  }
  if (take.state === 'choose_track') {
    head.append(discard);
    box.append(renderTrackChooser(take));
    return box;
  }
  // Several speakers and AI identification on: clips are reviewed per character, not per file
  if (take.state === 'done' && isCloneTake(take)) {
    box.classList.add('take--compact');
    head.append(take.attributed
      ? el('span', { className: 'pill pill--ok', textContent: 'Lines identified', title: 'Pick a character above to review their clips' })
      : el('span', { className: 'pill', textContent: people.identify?.running ? 'AI reading…' : 'Waiting for AI' }),
    discard);
    return box;
  }

  const edits = takeEditsFor(take);
  const rerender = () => { const node = renderTake(take); box.replaceWith(node); };
  const visible = (i) => !take.speakers || edits.speakers.has(take.segments[i].speaker);
  const kept = () => take.segments.filter((_, i) => visible(i) && edits.clips[i]?.keep !== false).length;
  const save = el('button', { type: 'button', className: 'btn btn--primary' });
  const saveBottom = el('button', { type: 'button', className: 'btn btn--primary' });
  const updateSave = () => {
    for (const b of [save, saveBottom]) {
      b.textContent = `✓ Save ${kept()} clips to the dataset`;
      b.disabled = kept() === 0;
    }
  };

  const noise = el('select', {},
    new Option('Noise: keep', 'off'), new Option('Noise: reduce', 'light'), new Option('Noise: remove', 'strong'));
  noise.value = take.denoise;
  noise.title = 'Background noise reduction for these clips (the original is kept)';
  noise.addEventListener('change', async () => {
    try {
      await api(voiceUrl(`/freeform/${take.id}`), {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ denoise: noise.value }),
      });
      take.denoise = noise.value;
    } catch (err) {
      SMT.showError(err.message);
      noise.value = take.denoise;
    }
  });
  head.append(noise, save, discard);

  if (!take.segments.length) {
    box.append(el('p', { className: 'hint', textContent: 'No speech was found in this take.' }));
    updateSave();
    return box;
  }

  // Diarized take: pick whose clips to keep
  if (take.speakers) {
    if (!take.speakers.length) {
      box.append(el('p', { className: 'hint', textContent: 'Nobody spoke long enough to train on.' }));
      updateSave();
      return box;
    }
    box.append(el('p', {
      className: 'hint',
      textContent: 'Who do you want? Play the samples and pick the speaker. If one person was split into two groups, pick both.',
    }));
    const grid = el('div', { className: 'speakers' });
    const bestSimilarity = Math.max(...take.speakers.map((s) => s.similarity ?? -1));
    for (const speaker of take.speakers) {
      const chosen = edits.speakers.has(speaker.id);
      const card = el('div', { className: `speaker${chosen ? ' selected' : ''}` });
      const person = speaker.person && personById(speaker.person);
      const head2 = el('div', { className: 'speaker-head' },
        person ? editableName(person) : el('strong', { textContent: displayName(speaker) }));
      if (person && person.id === people.target) {
        head2.append(el('span', { className: 'pill pill--ok', textContent: 'The voice I want' }));
      } else if (person && person.takes.length > 1) {
        const others = person.takes.length - 1;
        head2.append(el('span', { className: 'pill', textContent: `also in ${others} other file${others === 1 ? '' : 's'}` }));
      }
      if (speaker.similarity !== null) {
        const like = speaker.similarity === bestSimilarity && speaker.similarity >= SUGGEST_SIMILARITY;
        head2.append(el('span', {
          className: `pill${like ? ' pill--ok' : ''}`,
          textContent: like ? `Sounds like you · ${Math.round(speaker.similarity * 100)}%` : `${Math.round(speaker.similarity * 100)}% like you`,
          title: "Similarity to this voice's existing recordings",
        }));
      }
      card.append(head2, el('small', { className: 'hint', textContent: `${clock(speaker.seconds)} of speech · ${speaker.clips} clips` }));
      const quote = take.segments[speaker.samples[0]]?.text;
      if (quote) card.append(el('p', { className: 'quote', textContent: `“${quote}”` }));
      const samples = el('div', { className: 'samples' });
      speaker.samples.forEach((index, n) => {
        const play = el('button', { type: 'button', className: 'btn btn--secondary', textContent: `▶ ${n + 1}` });
        play.addEventListener('click', () => playClip(take, index, play));
        samples.append(play);
      });
      const pick = el('button', {
        type: 'button', className: `btn ${chosen ? 'btn--primary' : 'btn--secondary'}`,
        textContent: chosen ? '✓ Using this voice' : 'Use this voice',
      });
      pick.addEventListener('click', () => {
        if (chosen) edits.speakers.delete(speaker.id); else edits.speakers.add(speaker.id);
        rerender();
      });
      samples.append(pick);
      card.append(samples);
      grid.append(card);
    }
    box.append(grid);
    if (!edits.speakers.size) {
      box.append(el('p', { className: 'next-step', textContent: 'Next: pick the speaker you want above.' }));
      updateSave();
      return box;
    }
    box.append(el('p', {
      className: 'next-step',
      textContent: 'Next: check the clips below, then press ✓ Save to add them to the dataset.',
    }));
  }

  box.append(el('p', {
    className: 'hint',
    textContent: 'Check each clip: ▶ to listen, fix any wrong words (the text must match what was said exactly), untick clips with mistakes, music or other voices.',
  }));
  const list = el('div', { className: 'clips' });
  take.segments.forEach((segment, i) => {
    if (!visible(i)) return;
    const edit = (edits.clips[i] ||= { keep: true, text: segment.text });
    const row = el('div', { className: `clip${edit.keep ? '' : ' skipped'}` });
    const keep = el('input', { type: 'checkbox', checked: edit.keep, title: 'Keep this clip' });
    keep.addEventListener('change', () => {
      edit.keep = keep.checked;
      row.classList.toggle('skipped', !keep.checked);
      updateSave();
    });
    const play = el('button', { type: 'button', className: 'btn btn--secondary', textContent: '▶' });
    play.addEventListener('click', () => playClip(take, i, play));
    const text = el('input', { type: 'text', value: edit.text });
    text.addEventListener('input', () => { edit.text = text.value; });
    const dur = el('span', { className: 'dur', textContent: `${(segment.end - segment.start).toFixed(1)}s` });
    row.append(keep, play, text, dur);
    list.append(row);
  });
  box.append(list);
  box.append(el('div', { className: 'take-foot' }, saveBottom));
  updateSave();

  saveBottom.addEventListener('click', () => save.click());
  save.addEventListener('click', async () => {
    const clips = take.segments
      .map((_, i) => ({ index: i, text: (edits.clips[i]?.text ?? '').trim(), keep: edits.clips[i]?.keep !== false }))
      .filter((c, i) => visible(i) && c.keep && c.text);
    save.disabled = true;
    saveBottom.disabled = true;
    save.textContent = 'Saving…';
    saveBottom.textContent = 'Saving…';
    try {
      const result = await postJson(voiceUrl(`/freeform/${take.id}/save`), { clips });
      delete takeEdits[take.id];
      updateRecorded(result.recorded);
      box.replaceWith(el('div', { className: 'banner banner--success', textContent: `Added ${result.saved} clips to the dataset.` }));
      loadTakes();  // refresh the "waiting for review" counts and the people
    } catch (err) {
      SMT.showError(err.message);
      updateSave();
    }
  });
  return box;
}

function renderTrackChooser(take) {
  const wrap = el('div', { className: 'tracks' });
  wrap.append(el('p', {
    className: 'hint',
    textContent: 'This file has several audio tracks. Pick the one with the voice you want (the voice’s language is preselected).',
  }));
  let chosen = take.track;
  const dialogue = el('label', { className: 'switch' },
    el('input', { type: 'checkbox', checked: take.dialogue }),
    el('span', { textContent: 'Dialogue only: use the center channel of surround sound (cleaner voice, less music and effects)' }));
  const updateDialogue = () => {
    const surround = take.tracks[chosen].surround;
    dialogue.classList.toggle('hidden', !surround);
    dialogue.querySelector('input').checked = surround;
  };
  const list = el('div', { className: 'track-list' });
  take.tracks.forEach((track) => {
    const radio = el('input', { type: 'radio', name: `track-${take.id}`, checked: track.index === chosen });
    radio.addEventListener('change', () => { chosen = track.index; updateDialogue(); });
    const notes = [];
    if (track.index === take.track) notes.push('suggested');
    if (track.default) notes.push('default track');
    if (track.surround) notes.push('surround');
    list.append(el('label', { className: 'track' }, radio,
      el('span', { className: 'opt-main' }, el('strong', { textContent: trackLabel(track) }),
        el('small', { textContent: notes.join(' · ') || ' ' }))));
  });
  const go = el('button', { type: 'button', className: 'btn btn--primary', textContent: 'Use this track' });
  go.addEventListener('click', async () => {
    go.disabled = true;
    try {
      await postJson(voiceUrl(`/freeform/${take.id}/track`), {
        track: chosen, dialogue: dialogue.querySelector('input').checked,
      });
    } catch (err) {
      SMT.showError(err.message);
      go.disabled = false;
      return;
    }
    loadTakes();
  });
  wrap.append(list, dialogue, el('div', { className: 'row' }, go));
  updateDialogue();
  return wrap;
}

let clipButton = null;
async function playClip(take, index, button) {
  const audio = $('#clip-audio');
  const label = button.dataset.label || button.textContent;
  button.dataset.label = label;
  if (clipButton) clipButton.textContent = clipButton.dataset.label;
  if (clipButton === button && !audio.paused) { audio.pause(); clipButton = null; return; }
  clipButton = button;
  button.textContent = '■';
  audio.onended = () => { button.textContent = label; clipButton = null; };
  audio.src = voiceUrl(`/freeform/${take.id}/clips/${index}.wav?denoise=${take.denoise}`);
  await SMT.applyOutput(audio);
  audio.play().catch(() => { button.textContent = label; });
}

$('#upload-btn').addEventListener('click', async () => {
  const file = $('#upload-input').files[0];
  if (!file) {
    return;
  }
  const form = new FormData();
  form.set('dataset', file);
  $('#upload-btn').disabled = true;
  $('#upload-btn').textContent = 'Uploading…';
  try {
    const result = await api(voiceUrl('/upload'), { method: 'POST', body: form });
    updateRecorded(result.recorded);
    alert(`Added ${result.imported} clips to the dataset.`);
  } catch (err) {
    SMT.showError(`Upload failed: ${err.message}`);
  } finally {
    $('#upload-btn').disabled = false;
    $('#upload-btn').textContent = 'Upload';
  }
});

// ---------------------------------------------------------------------------
// 3. Train

function renderPresets() {
  const box = $('#presets');
  box.innerHTML = '';
  Object.entries(info.presets).forEach(([key, preset]) => {
    const button = document.createElement('button');
    button.className = 'preset';
    button.dataset.preset = key;
    button.innerHTML = '<strong></strong><span></span>';
    button.querySelector('strong').textContent = preset.label;
    button.querySelector('span').textContent = preset.hint;
    button.addEventListener('click', () => selectPreset(key));
    box.appendChild(button);
  });
}

function selectPreset(key) {
  selectedPreset = key;
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('selected', b.dataset.preset === key));
  if (key !== 'custom' && info.presets[key]) {
    $('#hours-input').value = info.presets[key].hours;
  }
}

$('#hours-input').addEventListener('input', () => {
  const hours = Number($('#hours-input').value);
  const match = Object.entries(info.presets).find(([, p]) => p.hours === hours);
  selectedPreset = match ? match[0] : 'custom';
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('selected', b.dataset.preset === selectedPreset));
});

function fillCheckpoints(defaults) {
  const select = $('#checkpoint-select');
  select.innerHTML = '';
  if (voice.training.hasCheckpoint) {
    select.add(new Option("Continue this voice's training", 'latest'));
  }
  if (voice.startingVoices.length) {
    select.add(new Option('Auto-detect: closest to your recordings', 'auto'));
  }
  const groups = Object.entries(info.checkpoints).sort(([a], [b]) => {
    const rank = (g) => (voice.checkpointGroups.includes(g) ? 0 : g === 'generic' ? 1 : 2);
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  groups.forEach(([group, entries]) => {
    const optgroup = document.createElement('optgroup');
    optgroup.label = group === 'generic' ? 'Any language' : group;
    entries.forEach((entry) => optgroup.appendChild(new Option(`${entry.name} · ${entry.gender}`, entry.url)));
    select.appendChild(optgroup);
  });
  select.add(new Option('Custom path or URL…', '__custom__'));
  select.add(new Option('Nothing (train from scratch — very slow)', ''));

  const known = Array.from(select.options).some((o) => o.value === defaults.checkpoint);
  select.value = known ? defaults.checkpoint : '__custom__';
  $('#checkpoint-input').value = known ? '' : defaults.checkpoint;
  show($('#checkpoint-input'), select.value === '__custom__');
  renderStartOptions();
}

$('#checkpoint-select').addEventListener('change', () => {
  show($('#checkpoint-input'), $('#checkpoint-select').value === '__custom__');
  renderStartOptions();
});

// ---- Starting voice picker (mirrors the "Start from" select) --------------------

let matchState = null;   // GET api/voices/{name}/match
let matchTimer = null;

let startListOpen = false;  // the pretrained voices, when Auto-detect is chosen

function chooseStart(value) {
  if (value === 'auto') startListOpen = false;  // Auto-detect: the other choices fold away
  $('#checkpoint-select').value = value;
  show($('#checkpoint-input'), false);
  renderStartOptions();
}

function matchSummary() {
  const min = matchState?.minRecordings ?? 5;
  const suggested = voice.startingVoices.find((v) => v.url === voice.suggested);
  const fallback = suggested ? suggested.name : 'the default voice';
  if (matchState?.state === 'running') return matchState.detail || 'Comparing your recordings…';
  if (matchState?.state === 'error') return `Could not compare: ${matchState.error}`;
  const match = matchState?.match;
  if (match && match.results.length) {
    const best = match.results[0];
    const more = voice.recorded - match.recordings;
    const since = more > 0 ? `; ${more} new since` : '';
    return `Best match so far: ${best.name} (${Math.round(best.score * 100)}% similar, from ${match.used} recordings${since}). Checked again when training starts.`;
  }
  if (voice.recorded < min) {
    return `Picks the closest voice when training starts. Needs ${min}+ recordings (you have ${voice.recorded}); until then it uses ${fallback}.`;
  }
  return 'Compares your recordings with the voices below when training starts. “Find match” shows the ranking now.';
}

function startRow(value, title, subtitle, extras = []) {
  const row = document.createElement('label');
  row.className = 'start-option';
  const radio = document.createElement('input');
  radio.type = 'radio';
  radio.name = 'start-voice';
  radio.value = value;
  radio.checked = $('#checkpoint-select').value === value;
  row.classList.toggle('selected', radio.checked);
  radio.addEventListener('change', () => chooseStart(value));
  const main = document.createElement('span');
  main.className = 'opt-main';
  const strong = document.createElement('strong');
  strong.textContent = title;
  const small = document.createElement('small');
  small.textContent = subtitle;
  main.append(strong, small);
  row.append(radio, main, ...extras);
  return row;
}

function renderStartOptions() {
  const box = $('#start-options');
  if (!voice || !box) return;
  box.innerHTML = '';

  if (voice.training.hasCheckpoint) {
    box.append(startRow('latest', "Continue this voice's training", 'Picks up where the last run stopped'));
  }
  if (!voice.startingVoices.length) {
    const none = document.createElement('p');
    none.className = 'hint';
    none.textContent = 'No pretrained voices for this language; choose one in Advanced settings.';
    box.append(none);
    return;
  }

  const running = matchState?.state === 'running';
  const findBtn = document.createElement('button');
  findBtn.type = 'button';
  findBtn.className = 'btn btn--secondary';
  findBtn.textContent = running ? 'Comparing…' : (matchState?.match ? 'Check again' : 'Find match');
  findBtn.disabled = running || voice.recorded < (matchState?.minRecordings ?? 5);
  findBtn.addEventListener('click', (e) => { e.preventDefault(); findMatch(); });
  const auto = startRow('auto', 'Auto-detect (recommended)', matchSummary(), [findBtn]);
  auto.classList.add('auto');
  box.append(auto);

  // With Auto-detect, the pretrained voices stay folded until asked for
  if ($('#checkpoint-select').value === 'auto') {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'btn btn--ghost start-toggle';
    toggle.textContent = startListOpen
      ? 'Hide the voices ▴'
      : `Choose a voice myself (${voice.startingVoices.length} voices, ▶ to hear them) ▾`;
    toggle.addEventListener('click', (e) => { e.preventDefault(); startListOpen = !startListOpen; renderStartOptions(); });
    box.append(toggle);
    if (!startListOpen) return;
  }

  const results = matchState?.match?.results || [];
  const scores = Object.fromEntries(results.map((r) => [r.url, r.score]));
  const bestUrl = results[0]?.url;
  const voices = [...voice.startingVoices].sort((a, b) => (
    (scores[b.url] ?? -1) - (scores[a.url] ?? -1)
    || (a.gender !== voice.gender) - (b.gender !== voice.gender)
    || a.name.localeCompare(b.name)
  ));
  for (const v of voices) {
    const play = document.createElement('button');
    play.type = 'button';
    play.className = 'btn btn--secondary';
    play.textContent = '▶';
    play.title = `Hear ${v.name}`;
    play.addEventListener('click', (e) => { e.preventDefault(); playSample(v.sample, play); });

    const score = document.createElement('span');
    score.className = `score${v.url === bestUrl ? ' best' : ''}`;
    score.textContent = v.url in scores ? `${Math.round(scores[v.url] * 100)}%` : '';
    score.title = 'Similarity to your recordings';

    const notes = [v.gender];
    if (v.url === bestUrl) notes.push('closest match');
    else if (v.url === voice.suggested && !results.length) notes.push('default pick');
    box.append(startRow(v.url, v.name, notes.join(' · '), [score, play]));
  }
}

let sampleButton = null;
function playSample(url, button) {
  const audio = $('#sample-audio');
  const reset = () => { if (sampleButton) sampleButton.textContent = '▶'; sampleButton = null; };
  if (sampleButton === button && !audio.paused) {
    audio.pause();
    reset();
    return;
  }
  reset();
  sampleButton = button;
  button.textContent = '■';
  audio.onended = reset;
  audio.onerror = () => { button.textContent = 'no sample'; button.disabled = true; sampleButton = null; };
  audio.src = url;
  SMT.applyOutput(audio).then(() => audio.play()).catch(() => {});
}

async function loadMatch() {
  clearTimeout(matchTimer);
  if (!voice) return;
  const name = voice.name;
  try {
    const state = await api(voiceUrl('/match'));
    if (!voice || voice.name !== name) return;
    matchState = state;
  } catch (err) {
    matchState = null;
  }
  renderStartOptions();
  if (matchState?.state === 'running') matchTimer = setTimeout(loadMatch, 2000);
}

async function findMatch() {
  try {
    matchState = await postJson(voiceUrl('/match'), {});
  } catch (err) {
    SMT.showError(err.message);
    return;
  }
  renderStartOptions();
  matchTimer = setTimeout(loadMatch, 1500);
}

function fillTrainForm(defaults) {
  fillCheckpoints(defaults);
  $('#hours-input').value = defaults.hours;
  $('#epochs-input').value = defaults.epochs;
  $('#batch-input').value = defaults.batch_size;
  $('#device-select').value = defaults.accelerator;
  $('#rate-select').value = String(defaults.sample_rate);
  selectPreset(defaults.preset);
}

function trainSettings() {
  const choice = $('#checkpoint-select').value;
  return {
    preset: selectedPreset,
    hours: Number($('#hours-input').value) || 0,
    epochs: Number($('#epochs-input').value) || 0,
    checkpoint: choice === '__custom__' ? $('#checkpoint-input').value.trim() : choice,
    batch_size: Number($('#batch-input').value) || 0,
    accelerator: $('#device-select').value,
    sample_rate: Number($('#rate-select').value),
  };
}

$('#train-btn').addEventListener('click', async () => {
  if (voice.recorded < 10) {
    SMT.showError(pendingTakes
      ? `The dataset has ${voice.recorded} clips: the clips from your recorded or imported takes aren't saved yet. Review them in 2. Dataset and press ✓ Save, then train.`
      : `The dataset has ${voice.recorded} clips; at least 10 are needed (50+ recommended). Add clips in 2. Dataset first.`);
    if (pendingTakes) goToReview();
    return;
  }
  if (voice.recorded < 50 && !confirm(
    `Only ${voice.recorded} clips in the dataset${pendingClips ? ` (${pendingClips} more are waiting for review, not saved yet)` : ''}. The voice will sound rough. Train anyway?`,
  )) {
    return;
  }
  try {
    renderStatus(await postJson(voiceUrl('/train'), trainSettings()));
  } catch (err) {
    showError(err.message);
  }
});

$('#stop-btn').addEventListener('click', async () => {
  if (!confirm('Stop training? The latest version will be exported so you can still use it.')) {
    return;
  }
  renderStatus(await postJson(voiceUrl('/stop')));
});

function showError(text) {
  $('#train-error').textContent = text;
  show($('#train-error'), !!text);
}

function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s % 60).padStart(2, '0')}s`;
}

const STAGES = ['prepare', 'download', 'train', 'export'];
const STAGE_LABELS = {
  prepare: 'Preparing recordings…',
  download: 'Downloading base voice…',
  train: 'Training',
  export: 'Exporting voice…',
};

function renderStatus(s) {
  status = s;
  const running = s.state === 'running';
  const busyElsewhere = info.busy && info.busy !== voice.name;

  show($('#train-btn'), !running);
  show($('#stop-btn'), running);
  $('#train-btn').disabled = s.exporting || !!busyElsewhere || !info.device.ok;
  $('#train-btn').textContent = s.exports.length > 0 || s.hasCheckpoint ? 'Train more' : 'Train';
  document.querySelectorAll('#train-card input, #train-card select, .preset').forEach((el) => {
    el.disabled = running;
  });
  show($('#progress'), running || s.exporting || s.state !== 'idle');

  // Stage chips
  const current = STAGES.indexOf(s.stage);
  document.querySelectorAll('.stage').forEach((chip) => {
    const index = STAGES.indexOf(chip.dataset.stage);
    chip.classList.toggle('active', index === current);
    chip.classList.toggle('done', current >= 0 ? index < current : s.state === 'succeeded');
  });

  // Label + time bar
  const fill = $('#timebar-fill');
  const limit = (s.settings && s.settings.hours) ? s.settings.hours * 3600 : 0;
  const elapsed = s.started ? ((s.finished || s.now) - s.started) : 0;
  let label;
  if (running) {
    label = STAGE_LABELS[s.stage] || 'Working…';
    if (s.stage === 'train') {
      label += s.epoch !== null ? ` · epoch ${s.epoch}` : '';
      label += ` · ${formatDuration(elapsed)}${limit ? ` of ${formatDuration(limit)}` : ''}`;
    }
  } else if (s.exporting) {
    label = STAGE_LABELS.export;
  } else {
    label = {
      succeeded: `Finished after ${formatDuration(elapsed)}`,
      stopped: `Stopped after ${formatDuration(elapsed)}`,
      failed: 'Training failed',
      idle: '',
    }[s.state];
  }
  $('#progress-label').textContent = label;
  const timed = running && s.stage === 'train' && limit;
  fill.classList.toggle('indeterminate', (running || s.exporting) && !timed);
  fill.style.width = timed ? `${Math.min(100, (100 * elapsed) / limit)}%` : (s.state === 'idle' ? '0' : '100%');

  showError(s.state === 'failed' || s.error ? s.error || 'See the log for details.' : '');
  if (busyElsewhere && !running) {
    showError(`Another voice (${info.busy}) is training. Wait for it to finish or stop it first.`);
    // Check again until it's free (it may just be finishing after a stop). Status updates
    // come every second or so: don't restart a check that's already waiting.
    if (!infoTimer) infoTimer = setTimeout(refreshInfo, 5000);
  }

  renderExports(s);
}

function renderExports(s) {
  const exports = s.exports || [];
  show($('#export-area'), exports.length > 0);
  show($('#no-exports'), exports.length === 0);
  show($('#next-voices'), exports.length > 0 || Boolean(s.hasCheckpoint));
  show($('#next-lab'), exports.length > 0);
  show($('#delete-model'), exports.length > 0 || Boolean(s.hasCheckpoint));
  show($('#export-btn'), s.hasCheckpoint && (s.state === 'running' || exports.length === 0 || s.exporting));
  $('#export-btn').disabled = s.exporting;
  $('#export-btn').textContent = s.exporting ? 'Exporting…' : 'Export latest version now';

  const select = $('#export-select');
  const key = exports.map((e) => e.dir).join(',');
  if (select.dataset.key !== key) {
    const previous = select.value;
    const newest = exports.length > 0 ? exports[0].dir : '';
    select.innerHTML = '';
    exports.forEach((e, i) => {
      const when = new Date(e.created * 1000).toLocaleString();
      select.add(new Option(`Epoch ${e.epoch}${i === 0 ? ' (latest)' : ''} · ${when}`, e.dir));
    });
    // Jump to a newly exported version
    select.value = select.dataset.key && select.dataset.newest !== newest ? newest : (previous || newest);
    if (!select.value) {
      select.value = newest;
    }
    select.dataset.key = key;
    select.dataset.newest = newest;
    updateDownloads();
  }
}

function speed() {
  return Number($('#speed').value) || 100;
}

/** The model's file name at the chosen speed: en_US-tony_speed90-medium */
function speedName() {
  return speed() === 100 ? voice.modelName : voice.modelName.replace(/-medium$/, `_speed${speed()}-medium`);
}

function loadSpeed() {
  let saved = 100;
  try { saved = Number(localStorage.getItem(`speed:${voice.name}`)) || 100; } catch (e) { /* ignore */ }
  $('#speed').value = saved;
  showSpeed();
}

function showSpeed() {
  const s = speed();
  $('#speed-value').textContent = `${s}%`;
  $('#speed-hint').textContent = s === 100
    ? 'As trained. Slide left to slow it down; letting go of the slider speaks the text again at the new speed.'
    : `${s < 100 ? 'Slower' : 'Faster'} than trained. Downloads are named ${speedName()} and have this speed built into the model, so they sound the same in any app.`;
  $('#howto-name').textContent = speedName();
  show($('#howto-speed'), s !== 100);
  updateDownloads();
}

$('#speed').addEventListener('input', () => {
  try { localStorage.setItem(`speed:${voice.name}`, String(speed())); } catch (e) { /* ignore */ }
  showSpeed();
});
// On release, speak again at the new speed (the player only replays the last clip)
$('#speed').addEventListener('change', () => {
  const audio = $('#speak-audio');
  if (!audio.classList.contains('hidden') && $('#speak-input').value.trim() && !$('#speak-btn').disabled) {
    audio.pause();
    $('#speak-btn').click();
  }
});

function updateDownloads() {
  const dir = $('#export-select').value;
  const entry = (status?.exports || []).find((e) => e.dir === dir);
  if (!entry) {
    return;
  }
  const base = voiceUrl(`/exports/${dir}`);
  const query = speed() === 100 ? '' : `?speed=${speed()}`;
  $('#dl-zip').href = `${base}/home-assistant.zip${query}`;
  $('#dl-onnx').href = `${base}/${entry.model}${query}`;
  $('#dl-json').href = `${base}/${entry.model}.json${query}`;
}

$('#export-select').addEventListener('change', updateDownloads);
$('#export-btn').addEventListener('click', async () => {
  try {
    renderStatus(await postJson(voiceUrl('/export')));
  } catch (err) {
    showError(err.message);
  }
});

$('#speak-btn').addEventListener('click', async () => {
  SMT.unlockAudio($('#speak-audio'));  // phones: allow playing once the audio arrives
  const button = $('#speak-btn');
  button.disabled = true;
  button.textContent = 'Speaking…';
  try {
    const wav = await postJson(voiceUrl('/speak'), {
      text: $('#speak-input').value,
      export: $('#export-select').value,
      speed: speed(),
    });
    const audio = $('#speak-audio');
    audio.src = URL.createObjectURL(wav);
    SMT.applyDownloads(audio, (await SMT.serverSettings())?.audio?.allowDownloads ?? true);
    await SMT.applyOutput(audio);
    show(audio, true);
    audio.play();
  } catch (err) {
    SMT.showError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = '🔊 Speak';
  }
});
$('#speak-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    $('#speak-btn').click();
  }
});

// ---------------------------------------------------------------------------
// Live status

function connectEvents() {
  const name = voice.name;
  events = new EventSource(`api/voices/${encodeURIComponent(name)}/events?since=${logCount}`);
  events.onmessage = (e) => {
    if (!voice || voice.name !== name) {
      return;
    }
    const data = JSON.parse(e.data);
    const wasRunning = status && status.state === 'running';
    const wasExporting = status && status.exporting;
    logCount = data.logCount;
    if (data.lines.length > 0) {
      const panel = $('#log-panel');
      const atBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 30;
      panel.textContent = (panel.textContent + '\n' + data.lines.join('\n')).split('\n').slice(-1500).join('\n').trim();
      if (atBottom) {
        panel.scrollTop = panel.scrollHeight;
      }
    }
    // A stopped run still makes a usable voice from its last checkpoint: busy until that's done too
    if (wasRunning !== (data.state === 'running') || Boolean(wasExporting) !== Boolean(data.exporting)) {
      refreshInfo();
    }
    renderStatus(data);
  };
}

let infoTimer = null;

/** Which voice (if any) is training, fresh from the server; shown again right away. */
async function refreshInfo() {
  clearTimeout(infoTimer);
  infoTimer = null;
  try {
    info = await api('api/info');
  } catch (err) {
    return;
  }
  if (status && voice) renderStatus(status);
}

// ---------------------------------------------------------------------------
// Clips in the dataset (Build Dataset): listen, correct, delete; delete the dataset

const manageUi = { clips: [], edits: new Map(), deleted: new Set(), open: new Set(), search: '' };

function groupLabel(key, clips) {
  if (key.startsWith('freeform:')) {
    const take = lastTakes.find((t) => t.id === key.slice(9));
    return take ? (take.name || `Recording ${take.id}`) : `Earlier file (${key.slice(9)})`;
  }
  if (key === 'upload') return 'Imported dataset (.zip)';
  if (key === 'freeform') return 'Speak freely, imports and Character clone';
  return `Read sentences · ${key}`;
}

async function loadManager() {
  if (VIEW !== 'dataset' || !voice) return;
  Object.assign(manageUi, { clips: [], edits: new Map(), deleted: new Set() });
  try {
    manageUi.clips = (await api(voiceUrl('/dataset/clips'))).clips;
  } catch (err) {
    SMT.showError(err.message);
  }
  renderManager();
}

function renderManager() {
  const box = $('#manage-list');
  box.innerHTML = '';
  const needle = manageUi.search.trim().toLowerCase();
  const clips = manageUi.clips.filter((c) => !needle || c.text.toLowerCase().includes(needle)
    || (manageUi.edits.get(c.id) || '').toLowerCase().includes(needle));
  const total = manageUi.clips.reduce((sum, c) => sum + (c.seconds || 0), 0);
  $('#manage-count').textContent = `${manageUi.clips.length} clips${total ? ` · ${clock(total)}` : ''}`;
  if (!clips.length) {
    box.append(el('p', { className: 'hint', textContent: manageUi.clips.length ? 'No clips with those words.' : 'No clips yet: add some above.' }));
  }
  // Freeform clips grouped by the file they came from (an episode, a recording)
  const groups = new Map();
  clips.forEach((c) => {
    const stem = c.id.split('/')[1];
    const take = c.group === 'freeform' && stem.match(/^(.+)_\d{4}$/);
    const key = take ? `freeform:${take[1]}` : c.group;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  });
  const sortedGroups = [...groups.entries()]
    .sort((a, b) => groupLabel(a[0]).localeCompare(groupLabel(b[0]), undefined, { numeric: true }));
  // Play through: the clips listed, in the order shown
  manageUi.order = sortedGroups.flatMap(([key, list]) => list.map((clip) => ({ clip, from: groupLabel(key) })));
  sortedGroups.forEach(([key, list]) => {
      const details = el('details', { className: 'manage-group' });
      details.open = manageUi.open.has(key) || Boolean(needle);
      const label = el('span', { className: 'grow' });
      const describe = () => {
        const changed = list.filter((c) => manageUi.edits.has(c.id)).length;
        const gone = list.filter((c) => manageUi.deleted.has(c.id)).length;
        label.textContent = [groupLabel(key), `${list.length} clip${list.length === 1 ? '' : 's'}`,
          changed && `${changed} edited`, gone && `${gone} to delete`].filter(Boolean).join(' · ');
      };
      describe();
      const playGroup = el('button', { type: 'button', className: 'btn btn--ghost', textContent: '▶ Play', title: 'Play this group through' });
      playGroup.addEventListener('click', (e) => {
        e.preventDefault();  // not open/close the group
        e.stopPropagation();
        startPlayer(list.map((clip) => ({ clip, from: groupLabel(key) })));
      });
      details.append(el('summary', { className: 'manage-summary' }, label, playGroup));
      const fill = () => {
        if (details.dataset.filled) return;
        details.dataset.filled = '1';
        const rows = el('div', { className: 'clips' });
        list.forEach((clip) => rows.append(managerRow(clip, () => { describe(); updateManagerSave(); })));
        details.append(rows);
      };
      details.addEventListener('toggle', () => {
        if (details.open) { manageUi.open.add(key); fill(); } else manageUi.open.delete(key);
      });
      if (details.open) fill();
      box.append(details);
    });
  updateManagerSave();
}

function managerRow(clip, changed) {
  const row = el('div', { className: 'clip' });
  const play = el('button', { type: 'button', className: 'btn btn--secondary', textContent: '▶' });
  play.addEventListener('click', () => playDatasetClip(clip, play));
  const text = el('input', { type: 'text', value: manageUi.edits.get(clip.id) ?? clip.text });
  text.addEventListener('input', () => {
    if (text.value.trim() && text.value.trim() !== clip.text) manageUi.edits.set(clip.id, text.value.trim());
    else manageUi.edits.delete(clip.id);
    row.classList.toggle('clip--changed', manageUi.edits.has(clip.id));
    changed();
  });
  const remove = el('button', { type: 'button', className: 'btn btn--ghost', textContent: '🗑', title: 'Delete this clip (with Save changes)' });
  const mark = () => {
    const gone = manageUi.deleted.has(clip.id);
    row.classList.toggle('clip--deleted', gone);
    remove.textContent = gone ? '↩' : '🗑';
    remove.title = gone ? 'Keep this clip' : 'Delete this clip (with Save changes)';
    text.disabled = gone;
  };
  remove.addEventListener('click', () => {
    if (manageUi.deleted.has(clip.id)) manageUi.deleted.delete(clip.id); else manageUi.deleted.add(clip.id);
    mark();
    changed();
  });
  row.classList.toggle('clip--changed', manageUi.edits.has(clip.id));
  mark();
  row.append(play, text, el('span', { className: 'dur', textContent: clip.seconds ? `${clip.seconds.toFixed(1)}s` : '' }), remove);
  row.dataset.id = clip.id;
  // The player marks or corrects clips too: bring this row (and its group's counts) up to date
  row.sync = () => {
    text.value = manageUi.edits.get(clip.id) ?? clip.text;
    row.classList.toggle('clip--changed', manageUi.edits.has(clip.id));
    mark();
    changed();
  };
  return row;
}

// ---- Play through: every clip, one after another, with its words ------------------------

const player = { queue: [], index: 0, playing: false, timer: null, box: null, audio: null };

function playerDelay() {
  try { return Number(localStorage.getItem('voice.playDelay') ?? 1); } catch (e) { return 1; }
}

function buildPlayer() {
  const audio = el('audio');
  const delay = el('select', { title: 'Pause between clips' },
    ...[0, 0.5, 1, 2, 3, 5].map((s) => new Option(s ? `${s} s` : 'none', String(s))));
  delay.value = String(playerDelay());
  delay.addEventListener('change', () => {
    try { localStorage.setItem('voice.playDelay', delay.value); } catch (e) { /* ignore */ }
  });
  const box = el('div', { className: 'player hidden', tabIndex: -1 });
  const button = (text, title, onClick) => {
    const b = el('button', { type: 'button', className: 'btn btn--secondary', textContent: text, title });
    b.addEventListener('click', () => {
      onClick();
      // Keys go to the player again (Space shouldn't also "click" this button)
      if (!box.querySelector('.player-edit:focus')) box.focus();
    });
    return b;
  };
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'Play through the dataset');
  const screen = el('div', { className: 'player-screen' },
    el('div', { className: 'player-from' }),
    el('div', { className: 'player-subtitle' }),
    el('input', { type: 'text', className: 'player-edit hidden' }));
  const bar = el('div', { className: 'player-bar' }, el('div', { className: 'player-fill' }));
  const controls = el('div', { className: 'player-controls' },
    button('⏮', 'Previous clip (←)', () => playerGo(player.index - 1)),
    button('⏸', 'Pause / play (Space)', () => playerToggle()),
    button('⏭', 'Next clip (→)', () => playerGo(player.index + 1)),
    el('span', { className: 'player-pos' }),
    el('label', { className: 'player-delay' }, 'Pause between clips ', delay),
    el('span', { className: 'grow' }),
    button('🗑 Mark', 'Mark this clip for deletion, with Save changes (D)', () => playerMark()),
    button('✎ Fix', 'Correct the words: pauses; Enter keeps the change and plays on, Esc cancels (F)', () => playerEdit()),
    button('✕', 'Close (Esc)', () => stopPlayer()));
  const keys = el('div', { className: 'player-keys hint',
    textContent: 'Space pause/play · ← → previous/next · D mark for deletion · F fix the words · Esc close' });
  box.append(screen, bar, controls, keys, audio);
  document.body.append(box);
  audio.addEventListener('ended', () => {
    if (!player.playing) return;
    clearTimeout(player.timer);
    player.timer = setTimeout(() => playerGo(player.index + 1), Number(delay.value) * 1000);
  });
  audio.addEventListener('timeupdate', () => {
    const part = audio.duration ? audio.currentTime / audio.duration : 0;
    box.querySelector('.player-fill').style.width = `${((player.index + part) / player.queue.length) * 100}%`;
  });
  audio.addEventListener('error', () => { if (player.playing) playerGo(player.index + 1); });  // skip unreadable clips
  const edit = screen.querySelector('.player-edit');
  edit.addEventListener('keydown', (e) => {
    // Done editing: the keys go back to the player
    if (e.key === 'Enter') { e.preventDefault(); playerSaveEdit(); box.focus(); playerToggle(true); }
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      edit.classList.add('hidden');
      screen.querySelector('.player-subtitle').classList.remove('hidden');
      box.focus();
    }
  });
  edit.addEventListener('blur', () => playerSaveEdit());
  player.box = box;
  player.audio = audio;
}

function startPlayer(queue) {
  const list = queue.filter(({ clip }) => !manageUi.deleted.has(clip.id));
  if (!list.length) return;
  if (!player.box) buildPlayer();
  player.queue = list;
  show(player.box, true);
  document.body.classList.add('player-open');
  player.box.focus();
  playerGo(0);
}

function stopPlayer() {
  if (!player.box) return;
  clearTimeout(player.timer);
  player.playing = false;
  player.audio.pause();
  show(player.box, false);
  document.body.classList.remove('player-open');
  document.querySelectorAll('#manage-list .clip--playing').forEach((r) => r.classList.remove('clip--playing'));
}

function playerRow(clip) {
  return document.querySelector(`#manage-list .clip[data-id="${CSS.escape(clip.id)}"]`);
}

function playerGo(index) {
  clearTimeout(player.timer);
  if (index < 0) index = 0;
  if (index >= player.queue.length) {  // the end
    player.playing = false;
    player.box.querySelector('.player-controls button:nth-child(2)').textContent = '▶';
    player.box.querySelector('.player-subtitle').textContent = `Done: ${player.queue.length} clips.`;
    player.box.querySelector('.player-fill').style.width = '100%';
    return;
  }
  player.index = index;
  const { clip, from } = player.queue[index];
  const box = player.box;
  box.querySelector('.player-from').textContent = from;
  const subtitle = box.querySelector('.player-subtitle');
  subtitle.textContent = manageUi.edits.get(clip.id) ?? clip.text;
  subtitle.classList.remove('hidden');
  box.querySelector('.player-edit').classList.add('hidden');
  box.classList.toggle('player--marked', manageUi.deleted.has(clip.id));
  box.querySelector('.player-pos').textContent = `${index + 1} of ${player.queue.length}`;
  document.querySelectorAll('#manage-list .clip--playing').forEach((r) => r.classList.remove('clip--playing'));
  const row = playerRow(clip);
  if (row) {
    row.classList.add('clip--playing');
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  const [group, stem] = clip.id.split('/');
  player.audio.src = voiceUrl(`/dataset/audio/${encodeURIComponent(group)}/${encodeURIComponent(stem)}`);
  player.playing = true;
  box.querySelector('.player-controls button:nth-child(2)').textContent = '⏸';
  SMT.applyOutput(player.audio).then(() => player.audio.play()).catch(() => {});
}

function playerToggle(forcePlay) {
  if (!player.box) return;
  const play = forcePlay ?? !player.playing;
  const button = player.box.querySelector('.player-controls button:nth-child(2)');
  if (play) {
    if (player.index >= player.queue.length || player.audio.ended) { playerGo(player.index + (player.audio.ended ? 1 : 0)); return; }
    player.playing = true;
    button.textContent = '⏸';
    player.audio.play().catch(() => {});
  } else {
    player.playing = false;
    clearTimeout(player.timer);
    button.textContent = '▶';
    player.audio.pause();
  }
}

function playerMark() {
  const entry = player.queue[player.index];
  if (!entry) return;
  const { clip } = entry;
  if (manageUi.deleted.has(clip.id)) manageUi.deleted.delete(clip.id); else manageUi.deleted.add(clip.id);
  player.box.classList.toggle('player--marked', manageUi.deleted.has(clip.id));
  playerRow(clip)?.sync?.();
  updateManagerSave();
}

function playerEdit() {
  const entry = player.queue[player.index];
  if (!entry) return;
  playerToggle(false);
  const edit = player.box.querySelector('.player-edit');
  edit.value = manageUi.edits.get(entry.clip.id) ?? entry.clip.text;
  edit.dataset.id = entry.clip.id;
  player.box.querySelector('.player-subtitle').classList.add('hidden');
  edit.classList.remove('hidden');
  edit.focus();
}

function playerSaveEdit() {
  const edit = player.box.querySelector('.player-edit');
  if (edit.classList.contains('hidden')) return;
  const entry = player.queue.find(({ clip }) => clip.id === edit.dataset.id);
  if (entry) {
    const value = edit.value.trim();
    if (value && value !== entry.clip.text) manageUi.edits.set(entry.clip.id, value);
    else manageUi.edits.delete(entry.clip.id);
    player.box.querySelector('.player-subtitle').textContent = manageUi.edits.get(entry.clip.id) ?? entry.clip.text;
    playerRow(entry.clip)?.sync?.();
    updateManagerSave();
  }
  edit.classList.add('hidden');
  player.box.querySelector('.player-subtitle').classList.remove('hidden');
}

document.addEventListener('keydown', (e) => {
  if (!player.box || player.box.classList.contains('hidden') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof Element && e.target.matches('input, textarea, select')) return;
  if (e.key === ' ') { e.preventDefault(); playerToggle(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); playerGo(player.index + 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); playerGo(player.index - 1); }
  else if (e.key === 'd' || e.key === 'D' || e.key === 'Delete') { e.preventDefault(); playerMark(); }
  else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); playerEdit(); }
  else if (e.key === 'Escape') { e.preventDefault(); stopPlayer(); }
});

$('#manage-play').addEventListener('click', () => startPlayer(manageUi.order || []));

function playDatasetClip(clip, button) {
  const audio = $('#clip-audio');
  const label = '▶';
  if (clipButton) clipButton.textContent = clipButton.dataset.label || label;
  if (clipButton === button && !audio.paused) { audio.pause(); clipButton = null; return; }
  clipButton = button;
  button.dataset.label = label;
  button.textContent = '■';
  audio.onended = () => { button.textContent = label; clipButton = null; };
  const [group, stem] = clip.id.split('/');
  audio.src = voiceUrl(`/dataset/audio/${encodeURIComponent(group)}/${encodeURIComponent(stem)}`);
  SMT.applyOutput(audio).then(() => audio.play()).catch(() => { button.textContent = label; });
}

function updateManagerSave() {
  const edits = [...manageUi.edits.keys()].filter((id) => !manageUi.deleted.has(id)).length;
  const gone = manageUi.deleted.size;
  const button = $('#manage-save');
  button.disabled = !edits && !gone;
  button.textContent = edits || gone
    ? `Save changes (${[edits && `${edits} edited`, gone && `${gone} deleted`].filter(Boolean).join(', ')})`
    : 'Save changes';
}

$('#manage-search').addEventListener('input', (e) => { manageUi.search = e.target.value; renderManager(); });

$('#manage-save').addEventListener('click', async () => {
  const button = $('#manage-save');
  button.disabled = true;
  button.textContent = 'Saving…';
  try {
    const result = await postJson(voiceUrl('/dataset/edit'), {
      clips: [...manageUi.edits.entries()].filter(([id]) => !manageUi.deleted.has(id)).map(([id, text]) => ({ id, text })),
      delete: [...manageUi.deleted],
    });
    updateRecorded(result.recorded);
    $('#manage-count').textContent = `Saved: ${result.edited} edited, ${result.deleted} deleted`;
  } catch (err) {
    SMT.showError(err.message);
  }
  loadManager();
  loadTakes();  // Character clone review and Files table follow
});

$('#delete-dataset').addEventListener('click', async () => {
  const trained = voice.training.exports.length > 0 || voice.training.hasCheckpoint;
  if (!confirm(trained
    ? `Delete the dataset "${voice.name}"? All ${voice.recorded} clips and the imported files waiting for review go. Its trained voice model stays (in Voices).`
    : `Delete the dataset "${voice.name}"? All ${voice.recorded} clips and the imported files go. It has no trained model, so the whole voice goes.`)) return;
  try {
    const result = await api(voiceUrl('/dataset'), { method: 'DELETE' });
    if (result.voiceDeleted) {
      rememberVoice('');
      await loadVoices();
    } else {
      await selectVoice(voice.name);
    }
  } catch (err) {
    SMT.showError(err.message);
  }
});

// Voices: delete one version, or the whole model
$('#delete-export').addEventListener('click', async () => {
  const dir = $('#export-select').value;
  if (!dir || !confirm(`Delete version ${dir.replace('epoch_', 'epoch ')} of "${voice.name}"?`)) return;
  try {
    await api(voiceUrl(`/exports/${dir}`), { method: 'DELETE' });
  } catch (err) {
    SMT.showError(err.message);
  }
  await refreshVoiceList();
  if ($('#voice-select').value) await selectVoice(voice.name); else await loadVoices();
});

$('#delete-model').addEventListener('click', async () => {
  if (!confirm(`Delete the voice model "${voice.name}": every version and its training runs? The dataset stays (Build Dataset), so you can train it again.`)) return;
  try {
    const result = await api(voiceUrl('/model'), { method: 'DELETE' });
    if (result.voiceDeleted) rememberVoice('');
  } catch (err) {
    SMT.showError(err.message);
    return;
  }
  await loadVoices();
});

// ---------------------------------------------------------------------------

async function main() {
  info = await api('api/info');
  const languageSelect = $('#new-language');
  info.languages.forEach((l) => languageSelect.add(new Option(l.name, l.code)));
  const browserLanguage = (navigator.language || 'en-US').toLowerCase();
  const match = info.languages.find((l) => l.code.toLowerCase() === browserLanguage)
    || info.languages.find((l) => l.code.toLowerCase().startsWith(browserLanguage.split('-')[0]));
  if (match) {
    languageSelect.value = match.code;
  }
  info.accelerators.forEach((a) => $('#device-select').add(new Option(a, a)));
  renderPresets();

  const banner = $('#device-banner');
  if (!info.device.ok) {
    banner.textContent = 'Piper training is not installed, so you can record but not train. '
      + 'Run the toolkit with its Docker image (see README).';
    show(banner, true);
  } else if (!info.device.gpu) {
    banner.textContent = 'No NVIDIA GPU found. Training will run on the CPU and be very slow '
      + '(expect days, not hours, for a good voice).';
    show(banner, true);
  }

  if (!micAvailable()) {
    showMicProblem('Browsers only allow the microphone on http://localhost or HTTPS. '
      + 'Open this page through an SSH tunnel (ssh -L 8765:localhost:8765 you@server, then '
      + 'http://localhost:8765), or upload recordings as a zip below.');
    $('#mic-test-btn').disabled = true;
    $('#mic-select').disabled = true;
  } else {
    listMicrophones();
  }

  await loadVoices();
}

main().catch((err) => {
  document.body.insertAdjacentHTML('afterbegin', '<div class="banner banner--error"></div>');
  document.querySelector('.banner--error').textContent = `Failed to load: ${err.message}`;
});

// Opening the log jumps to the newest lines
document.querySelector('.log-details').addEventListener('toggle', (e) => {
  if (e.target.open) {
    $('#log-panel').scrollTop = $('#log-panel').scrollHeight;
  }
});

// The Test Lab tab changed this voice's speed: follow it
window.addEventListener('storage', (e) => {
  if (voice && e.key === `speed:${voice.name}`) loadSpeed();
});
