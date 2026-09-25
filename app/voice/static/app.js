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

async function loadVoices(selectName) {
  const { voices } = await api('api/voices');
  const select = $('#voice-select');
  select.innerHTML = '<option value="">— choose a voice —</option>';
  voices.forEach((v) => {
    const option = new Option(`${v.name} · ${v.languageName} · ${v.recorded} recordings`, v.name);
    select.add(option);
  });

  let name = selectName;
  if (!name) {
    try { name = localStorage.getItem('voice'); } catch (e) { name = null; }
  }
  if (name && voices.some((v) => v.name === name)) {
    select.value = name;
  } else if (voices.length > 0) {
    select.value = voices[0].name;
  }

  show($('#new-voice-form'), voices.length === 0);
  await selectVoice(select.value);
}

async function selectVoice(name) {
  if (events) {
    events.close();
    events = null;
  }
  voice = name ? await api(`api/voices/${encodeURIComponent(name)}`) : null;
  try { localStorage.setItem('voice', name || ''); } catch (e) { /* ignore */ }

  [$('#record-card'), $('#train-card'), $('#test-card')].forEach((card) => show(card, !!voice));
  if (!voice) {
    $('#voice-summary').textContent = '';
    return;
  }

  $('#voice-summary').textContent =
    `${voice.languageName} · ${voice.gender} · phonemes: ${voice.espeak_voice} · model: ${voice.modelName}.onnx`;
  $('#howto-name').textContent = voice.modelName;
  $('#speak-input').value = TEST_SENTENCES[voice.language.split('-')[0]] || '';

  skip = 0;
  $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
  $('#record-status').textContent = mediaStream
    ? 'Read each sentence naturally in a quiet room. Press R to record, R again to stop.'
    : 'First pick your microphone and click “Enable microphone”. Use the same one every session.';
  updateRecorded(voice.recorded);
  await loadPrompt();
  if (micAvailable()) {
    await listMicrophones();
    checkMicMatch();
  }
  matchState = null;
  fillTrainForm(voice.defaults);
  loadMatch();
  $('#free-takes').innerHTML = '';
  Object.keys(renderedTakes).forEach((id) => delete renderedTakes[id]);
  loadTakes();
  logCount = 0;
  $('#log-panel').textContent = '';
  renderStatus(voice.training);
  connectEvents();
}

$('#voice-select').addEventListener('change', (e) => selectVoice(e.target.value));
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
  if (!['prompts', 'free', 'file'].includes(mode)) mode = 'prompts';
  recordMode = mode;
  document.querySelectorAll('.mode-btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  show($('#mic-block'), mode !== 'file');
  show($('#prompt-mode'), mode === 'prompts');
  show($('#free-mode'), mode === 'free');
  show($('#file-mode'), mode === 'file');
  show($('#takes-block'), mode !== 'prompts');  // takes waiting for review
  try { localStorage.setItem('voice.recordMode', mode); } catch (e) { /* ignore */ }
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
let importFiles = [];  // [{file, path, selected}]
let importSkipped = 0;

function formatSize(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

function setImportFiles(entries) {
  const media = entries.filter((e) => MEDIA_FILE.test(e.path) || e.file.type.startsWith('video/') || e.file.type.startsWith('audio/'));
  importSkipped = entries.length - media.length;
  importFiles = media
    .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }))
    .map((e) => ({ ...e, selected: true }));
  // Videos usually have several speakers (and a soundtrack)
  if (importFiles.some((e) => VIDEO_FILE.test(e.path) || e.file.type.startsWith('video/'))) {
    $('#free-diarize').checked = true;
    $('#file-denoise').value = 'strong';
  }
  renderImportList();
}

function renderImportList() {
  const box = $('#import-list');
  box.innerHTML = '';
  const chosen = importFiles.filter((e) => e.selected);
  $('#free-upload-btn').textContent = chosen.length > 1 ? `Import ${chosen.length} files` : 'Import';
  $('#free-upload-btn').disabled = chosen.length === 0;
  if (!importFiles.length) {
    if (importSkipped) box.append(el('p', { className: 'hint', textContent: `No audio or video files found (${importSkipped} other files skipped).` }));
    show(box, importSkipped > 0);
    return;
  }
  show(box, true);
  const total = chosen.reduce((sum, e) => sum + e.file.size, 0);
  const all = el('input', { type: 'checkbox', checked: chosen.length === importFiles.length, title: 'Select all' });
  all.addEventListener('change', () => { importFiles.forEach((e) => { e.selected = all.checked; }); renderImportList(); });
  box.append(el('div', { className: 'import-head' }, all, el('strong', {
    textContent: `${chosen.length} of ${importFiles.length} file${importFiles.length === 1 ? '' : 's'} · ${formatSize(total)}`
      + (importSkipped ? ` · ${importSkipped} other file${importSkipped === 1 ? '' : 's'} skipped` : ''),
  })));
  const list = el('div', { className: 'import-files' });
  importFiles.forEach((entry) => {
    const check = el('input', { type: 'checkbox', checked: entry.selected });
    check.addEventListener('change', () => { entry.selected = check.checked; renderImportList(); });
    list.append(el('label', { className: 'import-file' }, check,
      el('span', { className: 'import-path', textContent: entry.path, title: entry.path }),
      el('span', { className: 'dur', textContent: formatSize(entry.file.size) })));
  });
  box.append(list);
}

$('#free-file').addEventListener('change', () => {
  setImportFiles([...$('#free-file').files].map((file) => ({ file, path: file.name })));
});
$('#free-folder-btn').addEventListener('click', () => $('#free-folder').click());
$('#free-folder').addEventListener('change', () => {
  // webkitRelativePath keeps the folder structure: "Season 1/episode 03.mkv"
  setImportFiles([...$('#free-folder').files].map((file) => ({ file, path: file.webkitRelativePath || file.name })));
});

// Drag and drop files or folders onto the Import tab
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

const dropZone = $('#file-mode');
['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (e) => {
  e.preventDefault();
  dropZone.classList.add('dropping');
}));
['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, () => dropZone.classList.remove('dropping')));
dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  const items = [...(e.dataTransfer.items || [])];
  try {
    const entries = items.map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null)).filter(Boolean);
    const found = entries.length
      ? (await Promise.all(entries.map((entry) => readEntry(entry)))).flat()
      : [...e.dataTransfer.files].map((file) => ({ file, path: file.name }));
    setImportFiles(found);
  } catch (err) {
    SMT.showError(`Could not read the dropped files: ${err.message}`);
  }
});

$('#free-upload-btn').addEventListener('click', async () => {
  const chosen = importFiles.filter((e) => e.selected);
  if (!chosen.length) return;
  const button = $('#free-upload-btn');
  button.disabled = true;
  const diarize = $('#free-diarize').checked;
  const denoise = $('#file-denoise').value;
  let failed = 0;
  for (const [i, entry] of chosen.entries()) {
    const label = chosen.length > 1 ? `${i + 1} of ${chosen.length}: ${entry.path}` : entry.path;
    const ok = await uploadTake(entry.file, entry.file.name, diarize, denoise, entry.path, label);
    if (ok) entry.selected = false; else failed += 1;
    renderImportList();
    button.disabled = true;
  }
  importFiles = importFiles.filter((e) => e.selected);  // keep the failed ones to retry
  renderImportList();
  $('#free-file').value = '';
  $('#free-folder').value = '';
  $('#free-upload-status').textContent = failed
    ? `${failed} file${failed === 1 ? '' : 's'} could not be imported (still listed above to retry).`
    : (chosen.length > 1 ? `All ${chosen.length} files uploaded. They're processed one at a time, below.` : '');
});

function uploadTake(blob, filename, diarize, denoise, originalName = '', label = '') {
  const form = new FormData();
  form.set('audio', blob, filename);
  form.set('denoise', denoise);
  form.set('diarize', diarize ? 'true' : 'false');
  form.set('mic', micLabel || '');
  form.set('original_name', originalName);
  const status = $('#free-upload-status');
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
let lastTakes = [];
const peopleUi = { showAll: false, filter: '', selected: new Set() };
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
  const ids = new Set(people.people.map((p) => p.id));
  peopleUi.selected.forEach((id) => { if (!ids.has(id)) peopleUi.selected.delete(id); });
  renderPeople();
}

async function updatePerson(person, change) {
  try {
    people = await api(voiceUrl(`/speakers/${person.id}`), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change),
    });
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
      people = await postJson(voiceUrl(`/speakers/${id}/merge`), { into: intoId });
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
    people = await postJson(voiceUrl(`/speakers/${person.id}/not-same`), { other: otherId });
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
  Object.keys(renderedTakes).forEach((id) => { if (renderedTakes[id] === 'done') delete renderedTakes[id]; });
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
  actions.append(pick);
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
    row.append(el('span', { textContent: `Maybe the same as ${other.name} (${Math.round(hint.score * 100)}%)` }));
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

function renderPeople() {
  const box = $('#people-panel');
  box.innerHTML = '';
  const list = people.people.filter((p) => p.takes.length);
  show(box, list.length > 0);
  if (!list.length) return;

  const multi = lastTakes.filter((t) => t.speakers).length > 1;
  const suggestions = list.filter((p) => (p.similar || []).length).length;
  box.append(el('div', { className: 'people-head' },
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
  if (tools.childNodes.length) box.append(tools);

  const needle = peopleUi.filter.trim().toLowerCase();
  const sorted = [...list]
    .filter((p) => !needle || p.name.toLowerCase().includes(needle))
    .sort((a, b) => (b.id === people.target) - (a.id === people.target)
      || peopleUi.selected.has(b.id) - peopleUi.selected.has(a.id)
      || b.seconds - a.seconds);
  const shown = needle || peopleUi.showAll ? sorted : sorted.slice(0, PEOPLE_SHOWN);
  const grid = el('div', { className: 'people' });
  shown.forEach((person) => grid.append(personCard(person)));
  box.append(grid);
  if (shown.length < sorted.length) {
    const more = el('button', {
      type: 'button', className: 'btn btn--ghost',
      textContent: `Show all ${sorted.length} people (${sorted.length - shown.length} more with less speech)`,
    });
    more.addEventListener('click', () => { peopleUi.showAll = true; renderPeople(); });
    box.append(more);
  } else if (peopleUi.showAll && sorted.length > PEOPLE_SHOWN && !needle) {
    const less = el('button', { type: 'button', className: 'btn btn--ghost', textContent: 'Show fewer' });
    less.addEventListener('click', () => { peopleUi.showAll = false; renderPeople(); });
    box.append(less);
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
      box.append(el('div', { className: 'take-foot' },
        el('span', { className: 'hint', textContent: 'Review the clips in each file below first; your edits and unticks are kept.' }), saveAll));
    }
  }
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
  if (takes.some((t) => t.speakers)) await loadPeople(); else { people = { people: [], target: null }; renderPeople(); }

  const box = $('#free-takes');
  const ids = new Set(takes.map((t) => t.id));
  box.querySelectorAll('.take').forEach((node) => {
    if (!ids.has(node.dataset.id)) { node.remove(); delete renderedTakes[node.dataset.id]; }
  });
  takes.slice().reverse().forEach((take) => {
    const existing = box.querySelector(`.take[data-id="${take.id}"]`);
    if (existing && renderedTakes[take.id] === take.state && take.state !== 'running') return;
    const node = renderTake(take);
    if (existing) existing.replaceWith(node); else box.append(node);
    renderedTakes[take.id] = take.state;
  });
  if (recordMode === 'prompts' && takes.some((t) => ['done', 'choose_track'].includes(t.state))) {
    // Something is waiting for review: show it where it came from
    setMode(takes.some((t) => t.tracks && t.source && !/^source\.(webm|ogg|m4a)$/.test(t.source)) ? 'file' : 'free');
  }
  if (takes.some((t) => t.state === 'running')) takesTimer = setTimeout(loadTakes, 2000);
  updatePending(takes);
}

// Clips from recorded/imported takes only join the dataset once saved:
// say so next to the counter and in the Train step.
let pendingClips = 0;
let pendingTakes = 0;
function updatePending(takes) {
  const waiting = takes.filter((t) => ['done', 'choose_track', 'running'].includes(t.state));
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

function chooseStart(value) {
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
  }

  renderExports(s);
}

function renderExports(s) {
  const exports = s.exports || [];
  show($('#export-area'), exports.length > 0);
  show($('#no-exports'), exports.length === 0);
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

function updateDownloads() {
  const dir = $('#export-select').value;
  const entry = (status.exports || []).find((e) => e.dir === dir);
  if (!entry) {
    return;
  }
  const base = voiceUrl(`/exports/${dir}`);
  $('#dl-zip').href = `${base}/home-assistant.zip`;
  $('#dl-onnx').href = `${base}/${entry.model}`;
  $('#dl-json').href = `${base}/${entry.model}.json`;
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
    logCount = data.logCount;
    if (data.lines.length > 0) {
      const panel = $('#log-panel');
      const atBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 30;
      panel.textContent = (panel.textContent + '\n' + data.lines.join('\n')).split('\n').slice(-1500).join('\n').trim();
      if (atBottom) {
        panel.scrollTop = panel.scrollHeight;
      }
    }
    if (wasRunning !== (data.state === 'running')) {
      refreshInfo();
    }
    renderStatus(data);
  };
}

async function refreshInfo() {
  info = await api('api/info');
}

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
