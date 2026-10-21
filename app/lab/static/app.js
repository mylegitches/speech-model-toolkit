/* Test Lab: wake word → (question → AI) → spoken reply. Protocol: app/lab/session.py */

const $ = (sel) => document.querySelector(sel);

const STATES = {
  loading: ['Loading', 'pill--warn pill--live'],
  listening: ['Listening', 'pill--ok pill--live'],
  heard: ['Heard it!', 'pill--ok'],
  recording: ['Your turn', 'pill--warn pill--live'],
  transcribing: ['Transcribing', 'pill--warn pill--live'],
  thinking: ['Thinking', 'pill--warn pill--live'],
  speaking: ['Speaking', 'pill--ok pill--live'],
};

let options = null;
let ws = null;
let audioCtx = null;
let mediaStream = null;
let micNode = null;
let state = 'stopped';
let history = [];          // for typed questions when no session is running
let detectionTimer = null;

// ---- Setup ------------------------------------------------------------------------

async function loadOptions() {
  options = await SMT.request('api/options');

  const ww = $('#wakeword-select');
  const keepWw = ww.value || SMT_lab('wakeword');
  ww.innerHTML = '';
  options.wakewords.forEach((name) => ww.add(new Option(name.replace(/_/g, ' '), name)));
  if (options.wakewords.includes(keepWw)) ww.value = keepWw;
  $('#no-wakeword').classList.toggle('hidden', options.wakewords.length > 0);

  const voice = $('#voice-select');
  const keepVoice = voice.value || SMT_lab('voice');
  voice.innerHTML = '';
  options.voices.forEach((v) => voice.add(new Option(
    v.id === 'default' && !v.ready ? `${v.label} · downloads on first use` : v.label, v.id)));
  if (options.voices.some((v) => v.id === keepVoice)) voice.value = keepVoice;
  loadSpeed();

  const a = options.assistant;
  const pill = $('#mode-pill');
  if (a.mode === 'ai') {
    pill.className = 'pill pill--ok';
    pill.textContent = 'AI answers';
    $('#mode-text').textContent = `${a.provider} · ${a.model}. After the wake word, ask anything.`;
  } else {
    pill.className = 'pill';
    pill.textContent = 'Fixed reply';
    $('#mode-text').textContent = `Replies “${a.fixedReply}” (add an AI connection to ask questions).`;
  }

  showRepeatMode();
  renderPauses();
  renderScript();
  document.body.classList.toggle('no-downloads', !options.allowDownloads);
  document.querySelectorAll('.dl-btn').forEach((b) => b.classList.toggle('hidden', !options.allowDownloads));
  $('#start-btn').disabled = !ww.value && !ws;
}

function SMT_lab(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(`smt.lab.${key}`) || '';
    localStorage.setItem(`smt.lab.${key}`, value);
  } catch (e) { /* not remembered */ }
  return '';
}

$('#wakeword-select').addEventListener('change', (e) => {
  SMT_lab('wakeword', e.target.value);
  if (ws) { stop(); start(); }
});
/** Speed is stored per voice under the Voice tab's key (speed:<voice name>), so both tabs agree. */
function speedKey() {
  const id = $('#voice-select').value || 'default';
  return `speed:${id.startsWith('voice:') ? id.slice(6) : id}`;
}

function speed() {
  return Number($('#speed').value) || 100;
}

function loadSpeed() {
  let saved = 100;
  try { saved = Number(localStorage.getItem(speedKey())) || 100; } catch (e) { /* ignore */ }
  $('#speed').value = saved;
  $('#speed-value').textContent = `${speed()}%`;
}

$('#speed').addEventListener('input', () => {
  $('#speed-value').textContent = `${speed()}%`;
  try { localStorage.setItem(speedKey(), String(speed())); } catch (e) { /* ignore */ }
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'speed', speed: speed() }));
});

$('#voice-select').addEventListener('change', (e) => {
  SMT_lab('voice', e.target.value);
  loadSpeed();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'voice', voice: e.target.value, speed: speed() }));
  }
});

// ---- Session ----------------------------------------------------------------------

$('#start-btn').addEventListener('click', () => (ws ? stop() : start()));

async function start() {
  const wakeword = $('#wakeword-select').value;
  if (!wakeword) return;
  // Phones only allow audio started by a tap: prepare it before any await
  SMT.unlockAudio($('#player'));
  audioCtx = new AudioContext({ sampleRate: 16000 });
  SMT.keepAwake(true);
  await loadOptions();  // pick up Settings changes

  try {
    mediaStream = await SMT.openMic({
      sampleRate: 16000,
      echoCancellation: options.audio.echoCancellation,
      noiseSuppression: options.audio.noiseSuppression,
      autoGainControl: options.audio.autoGainControl,
    });
  } catch (err) {
    note(SMT.micError(err), true);
    stop();
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const base = location.pathname.replace(/[^/]*$/, '');
  ws = new WebSocket(`${proto}://${location.host}${base}api/session`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = async () => {
    ws.send(JSON.stringify({ type: 'start', wakeword, voice: $('#voice-select').value, speed: speed(), repeat: repeatOn() }));
    try {
      await startCapture();
    } catch (err) {
      note(SMT.micError(err), true);
      stop();
    }
  };
  ws.onmessage = (evt) => onMessage(JSON.parse(evt.data));
  ws.onclose = (evt) => {
    if (!ws) return;  // we stopped it
    // Closed by the server or the network, not by the Stop button
    if (evt.code !== 1000) {
      note(evt.code === 1006
        ? 'The connection to the server dropped. If this happens right away behind a reverse proxy, enable WebSocket support there.'
        : `The server ended the session${evt.reason ? `: ${evt.reason}` : ''}.`, true);
    }
    stop();
  };

  $('#start-btn').textContent = '⏹ Stop';
  $('#start-btn').classList.add('listening');
  const track = mediaStream.getAudioTracks()[0];
  if (track?.label) note(`Microphone: ${track.label}`);
}

async function startCapture() {
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate: 16000 });
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  await audioCtx.audioWorklet.addModule('../static/mic-processor.js');
  micNode = new AudioWorkletNode(audioCtx, 'mic-processor');
  micNode.port.onmessage = (evt) => {
    const int16 = new Int16Array(evt.data);
    let sum = 0;
    for (let i = 0; i < int16.length; i++) sum += (int16[i] / 32768) ** 2;
    updateVu(Math.sqrt(sum / int16.length));
    // Only send while the server listens: never the reply playing back
    if (ws && ws.readyState === WebSocket.OPEN && (state === 'listening' || state === 'recording')) {
      ws.send(evt.data);
    }
  };
  audioCtx.createMediaStreamSource(mediaStream).connect(micNode);
}

function stop() {
  const socket = ws;
  ws = null;
  if (socket && socket.readyState < 2) socket.close();
  if (micNode) { micNode.port.close(); micNode.disconnect(); micNode = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  $('#player').pause();
  SMT.keepAwake(false);
  setState('stopped', 'Press Start listening, then say your wake word.');
  $('#start-btn').textContent = '🎤 Start listening';
  $('#start-btn').classList.remove('listening');
  $('#start-btn').disabled = !$('#wakeword-select').value;
  updateVu(0);
  resetDetector();
}

function setState(name, detail = '') {
  state = name;
  const [label, cls] = STATES[name] || ['Stopped', ''];
  const pill = $('#state-pill');
  pill.className = `pill ${cls}`;
  pill.textContent = label;
  $('#state-detail').textContent = detail;
  $('#detector-label').textContent = name === 'stopped' ? '—' : label;
}

function onMessage(msg) {
  switch (msg.type) {
    case 'score':
      updateDetector(msg.score);
      break;
    case 'state':
      setState(msg.state, msg.detail);
      if (msg.state === 'heard') flashDetected();
      break;
    case 'transcript':
      addMessage('you', msg.text);
      break;
    case 'reply':
      addReply(msg.text, msg.audioUrl, true);
      break;
    case 'error':
      note(msg.message, true);
      break;
    default:
      break;
  }
}

// ---- Replies ----------------------------------------------------------------------

let pendingDone = null;  // finish callback of the reply currently playing

async function play(url, onDone) {
  const player = $('#player');
  // Replacing a reply that is still playing counts as finishing it
  const previous = pendingDone;
  pendingDone = null;
  player.onended = player.onerror = null;
  if (previous) previous();

  player.src = url;
  await SMT.applyOutput(player);
  let finished = false;
  const done = () => {
    if (finished) return;
    finished = true;
    pendingDone = null;
    player.onended = player.onerror = null;
    if (onDone) onDone();
  };
  pendingDone = done;
  player.onended = done;
  player.onerror = done;
  try {
    await player.play();
  } catch (err) {
    note(`Could not play audio: ${err.message}`, true);
    done();
  }
}

function playbackDone() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'playback_done' }));
}

function addReply(text, audioUrl, fromSession) {
  const msg = addMessage('bot', text);
  if (!audioUrl) {
    if (fromSession) playbackDone();
    return;
  }

  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  const replay = document.createElement('button');
  replay.className = 'btn btn--secondary';
  replay.textContent = '▶ Play';
  replay.addEventListener('click', () => play(audioUrl));
  actions.append(replay);

  const dl = document.createElement('a');
  dl.className = `btn btn--secondary dl-btn${options.allowDownloads ? '' : ' hidden'}`;
  dl.textContent = '⬇ .wav';
  dl.href = `${audioUrl}?download=true`;
  dl.setAttribute('download', '');
  actions.append(dl);
  msg.append(actions);

  play(audioUrl, fromSession ? playbackDone : null);
}

function addMessage(who, text) {
  $('#log .log-empty')?.remove();
  const msg = document.createElement('div');
  msg.className = `msg msg--${who}`;
  const label = document.createElement('span');
  label.className = 'msg-who';
  label.textContent = who === 'you' ? 'You' : 'Reply';
  const body = document.createElement('span');
  body.textContent = text;
  msg.append(label, body);
  $('#log').append(msg);
  $('#log').scrollTop = $('#log').scrollHeight;
  if (who === 'you') history.push({ role: 'user', content: text });
  if (who === 'bot') history.push({ role: 'assistant', content: text });
  history = history.slice(-12);
  return msg;
}

function note(text, isError = false) {
  $('#log .log-empty')?.remove();
  const msg = document.createElement('div');
  msg.className = `msg ${isError ? 'msg--error' : 'msg--note'}`;
  msg.textContent = text;
  $('#log').append(msg);
  $('#log').scrollTop = $('#log').scrollHeight;
}

$('#ask-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  SMT.unlockAudio($('#player'));
  const input = $('#ask-input');
  const text = input.value.trim();
  if (!text) return;

  if (ws && ws.readyState === WebSocket.OPEN) {
    if (state !== 'listening') return note('Wait until it is listening again, then send.', true);
    input.value = '';
    ws.send(JSON.stringify({ type: 'ask', text }));
    return;
  }

  input.value = '';
  const priorHistory = history.slice();
  addMessage('you', text);
  const button = e.target.querySelector('button');
  button.disabled = true;
  button.textContent = repeatOn() ? 'Speaking…' : 'Thinking…';
  try {
    const res = await fetch('api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: $('#voice-select').value, speed: speed(), history: priorHistory, repeat: repeatOn() }),
    });
    if (!res.ok) throw new Error(await SMT.errorMessage(res));
    const data = await res.json();
    if (data.error) note(data.error, true);
    addReply(data.reply, data.audioUrl, false);
  } catch (err) {
    note(err.message, true);
  } finally {
    button.disabled = false;
    button.textContent = 'Send';
  }
});

$('#reset-btn').addEventListener('click', () => {
  history = [];
  $('#log').innerHTML = '<p class="hint log-empty">New conversation. Say the wake word or type a question.</p>';
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'reset' }));
});

// ---- Meters -----------------------------------------------------------------------

function updateVu(rms) {
  const pct = Math.min(100, rms * 400);
  const bar = $('#vu-bar');
  bar.style.width = `${pct}%`;
  bar.classList.toggle('hot', pct > 60);
  bar.classList.toggle('clip', pct > 90);
  $('#vu-val').textContent = rms > 0 ? rms.toFixed(3) : '—';
}

function updateDetector(score) {
  $('#score-bar').style.width = `${Math.round(score * 100)}%`;
  $('#score-val').textContent = `score: ${score.toFixed(3)}`;
  if (!$('#detector-ring').classList.contains('active')) {
    $('#score-bar').classList.toggle('hot', score >= (options?.audio.threshold ?? 0.5));
  }
}

function flashDetected() {
  $('#detector-ring').classList.add('active');
  $('#score-bar').classList.add('hot');
  clearTimeout(detectionTimer);
  detectionTimer = setTimeout(() => {
    $('#detector-ring').classList.remove('active');
    $('#score-bar').classList.remove('hot');
  }, 1500);
}

function resetDetector() {
  clearTimeout(detectionTimer);
  $('#detector-ring').classList.remove('active');
  $('#score-bar').classList.remove('hot');
  $('#score-bar').style.width = '0%';
  $('#score-val').textContent = '';
}

// Refresh choices when the tab is shown again (new models, changed settings)
window.addEventListener('message', (e) => {
  if (e.origin === location.origin && e.data?.type === 'smt:shown' && !ws) loadOptions();
});

loadOptions().then(() => stop());

// The Voice tab changed this voice's speed: follow it
window.addEventListener('storage', (e) => {
  if (e.key === speedKey()) loadSpeed();
});


// ---- Repeat mode ------------------------------------------------------------------
// Speaks what you type or say, word for word, instead of asking the AI.

function repeatOn() { return $('#repeat-mode').checked; }

function showRepeatMode() {
  $('#repeat-mode').checked = SMT_lab('repeat') === '1';
  if (repeatOn()) {
    $('#mode-pill').className = 'pill pill--warn';
    $('#mode-pill').textContent = 'Repeat mode';
    $('#mode-text').textContent = 'Says back what you type or say after the wake word, word for word. No AI.';
  }
  $('#ask-input').placeholder = repeatOn() ? 'Type what the voice should say' : 'Type a question instead of speaking';
}

$('#repeat-mode').addEventListener('change', () => {
  SMT_lab('repeat', repeatOn() ? '1' : '0');
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'repeat', on: repeatOn() }));
  loadOptions();  // the mode line
});

// ---- Pauses -----------------------------------------------------------------------

let pauseSettings = null;
let pauseTimer = null;

function pauseRow(mark, value, onChange) {
  const out = document.createElement('span');
  out.className = 'value';
  const range = Object.assign(document.createElement('input'), { type: 'range', min: 0, max: mark.max, step: 25, value });
  const label = document.createElement('label');
  label.className = 'pause-row';
  const show = () => { out.textContent = Number(range.value) ? `${range.value} ms` : 'off'; };
  range.addEventListener('input', () => { show(); onChange(Number(range.value)); });
  label.append(document.createTextNode(mark.label), out, range);
  show();
  return label;
}

function renderPauses() {
  if (!options) return;
  pauseSettings = { ...options.pauses };
  $('#pause-enabled').checked = Boolean(pauseSettings.enabled);
  const box = $('#pause-sliders');
  box.innerHTML = '';
  options.pauseMarks.forEach((mark) => box.append(pauseRow(mark, pauseSettings[mark.key], (ms) => {
    pauseSettings[mark.key] = ms;
    savePauses();
  })));
  box.classList.toggle('pause-off', !pauseSettings.enabled);
}

function savePauses() {
  clearTimeout(pauseTimer);
  pauseTimer = setTimeout(async () => {
    try {
      options.pauses = await SMT.request('api/pauses', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pauseSettings),
      });
    } catch (err) {
      note(err.message, true);
    }
  }, 400);
}

$('#pause-enabled').addEventListener('change', (e) => {
  pauseSettings.enabled = e.target.checked;
  $('#pause-sliders').classList.toggle('pause-off', !pauseSettings.enabled);
  savePauses();
});

$('#pause-reset').addEventListener('click', () => {
  options.pauseMarks.forEach((mark) => { pauseSettings[mark.key] = mark.default; });
  options.pauses = { ...pauseSettings };
  renderPauses();
  savePauses();
});

const PREVIEW = 'Hello there. <break time="1s"/> I have something important to tell you... Are you ready, or should I wait? Okay!';

$('#pause-preview').addEventListener('click', async (e) => {
  const button = e.currentTarget;
  SMT.unlockAudio($('#player'));
  button.disabled = true;
  clearTimeout(pauseTimer);
  try {
    await SMT.request('api/pauses', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pauseSettings) });
    const data = await SMT.request('api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: PREVIEW, voice: $('#voice-select').value, speed: speed(), repeat: true }),
    });
    if (data.audioUrl) play(data.audioUrl);
  } catch (err) {
    note(err.message, true);
  } finally {
    button.disabled = false;
  }
});

// ---- Script: a conversation between characters ---------------------------------------
// Lines are kept in this browser. Each: {voice, text, gapAfter (ms), speed, pauses: {mark: ms}}

let script = [];
try { script = JSON.parse(localStorage.getItem('smt.lab.script') || '[]'); } catch (e) { script = []; }
const keepScript = () => { try { localStorage.setItem('smt.lab.script', JSON.stringify(script)); } catch (e) { /* not kept */ } };

function voiceSelect(value, onChange) {
  const select = document.createElement('select');
  options.voices.forEach((v) => select.add(new Option(v.label, v.id)));
  select.value = options.voices.some((v) => v.id === value) ? value : options.voices[0]?.id;
  select.addEventListener('change', () => onChange(select.value));
  return select;
}

function scriptLine(line, i) {
  const row = document.createElement('div');
  row.className = 'script-line';
  row.dataset.index = String(i);
  const voice = voiceSelect(line.voice, (v) => { line.voice = v; keepScript(); });
  line.voice = voice.value;
  const text = Object.assign(document.createElement('textarea'), { rows: 2, value: line.text || '', placeholder: 'What this character says' });
  text.addEventListener('input', () => { line.text = text.value; keepScript(); });
  const button = (label, title, onClick, cls = 'btn btn--ghost') => {
    const b = Object.assign(document.createElement('button'), { type: 'button', className: cls, textContent: label, title });
    b.addEventListener('click', onClick);
    return b;
  };
  const gap = pauseRow({ label: 'Pause after', max: 3000 }, line.gapAfter ?? 600, (ms) => { line.gapAfter = ms; keepScript(); });
  gap.classList.add('gap-after');
  // ⚙ this line's own pauses after punctuation (over the defaults in 4. Pauses)
  const timing = document.createElement('div');
  timing.className = 'line-timing hidden';
  const own = Object.assign(document.createElement('input'), { type: 'checkbox', checked: Boolean(line.pauses) });
  const sliders = document.createElement('div');
  sliders.className = 'pause-sliders';
  const fillTiming = () => {
    sliders.innerHTML = '';
    sliders.classList.toggle('pause-off', !line.pauses);
    options.pauseMarks.forEach((mark) => sliders.append(pauseRow(mark,
      line.pauses?.[mark.key] ?? options.pauses[mark.key], (ms) => {
        if (!line.pauses) return;
        line.pauses[mark.key] = ms;
        keepScript();
      })));
  };
  own.addEventListener('change', () => {
    line.pauses = own.checked ? { ...Object.fromEntries(options.pauseMarks.map((m) => [m.key, options.pauses[m.key]])), enabled: true } : null;
    keepScript();
    fillTiming();
  });
  const ownLabel = document.createElement('label');
  ownLabel.className = 'switch';
  ownLabel.append(own, Object.assign(document.createElement('span'), { textContent: 'This line has its own pauses (else the ones in 4. Pauses)' }));
  timing.append(ownLabel, sliders);
  const tools = document.createElement('span');
  tools.className = 'line-tools';
  tools.append(
    button('▶', 'Play this line', () => playScript([line], button)),
    button('⚙', 'This line’s pauses', () => { timing.classList.toggle('hidden'); if (!timing.classList.contains('hidden')) fillTiming(); }),
    button('↑', 'Move up', () => moveLine(i, -1)),
    button('↓', 'Move down', () => moveLine(i, 1)),
    button('🗑', 'Delete this line', () => { script.splice(i, 1); keepScript(); renderScript(); }),
  );
  row.append(Object.assign(document.createElement('span'), { className: 'line-no', textContent: String(i + 1) }),
    voice, text, tools, gap, timing);
  return row;
}

function moveLine(i, by) {
  const j = i + by;
  if (j < 0 || j >= script.length) return;
  [script[i], script[j]] = [script[j], script[i]];
  keepScript();
  renderScript();
}

function renderScript() {
  if (!options) return;
  if (!script.length) script = [{ voice: $('#voice-select').value, text: '', gapAfter: 600 }];
  const box = $('#script-lines');
  box.innerHTML = '';
  script.forEach((line, i) => box.append(scriptLine(line, i)));
}

$('#script-add').addEventListener('click', () => {
  // The next line: the other character of the last two, so a back-and-forth is quick to write
  const last = script[script.length - 1];
  const before = script[script.length - 2];
  const other = before && before.voice !== last?.voice ? before.voice
    : (options.voices.find((v) => v.id !== last?.voice) || options.voices[0]).id;
  script.push({ voice: other, text: '', gapAfter: last?.gapAfter ?? 600 });
  keepScript();
  renderScript();
  $('#script-lines').lastElementChild?.querySelector('textarea')?.focus();
});

$('#script-clear').addEventListener('click', () => {
  if (!confirm('Clear the whole script?')) return;
  script = [];
  keepScript();
  renderScript();
});

async function playScript(lines, button) {
  const player = $('#script-player');
  SMT.unlockAudio(player);
  const status = $('#script-status');
  status.textContent = 'Speaking…';
  try {
    const data = await SMT.request('api/script', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lines: lines.map((l) => ({ ...l, speed: Number(localStorage.getItem(`speed:${(l.voice || 'default').replace(/^voice:/, '')}`)) || 100 })) }),
    });
    status.textContent = '';
    const dl = $('#script-download');
    dl.href = `${data.audioUrl}?download=true`;
    dl.classList.toggle('hidden', !options.allowDownloads || lines !== script);
    player.src = data.audioUrl;
    await SMT.applyOutput(player);
    // Highlight the line being spoken
    const rows = [...$('#script-lines').children];
    const spoken = lines === script ? rows : rows.filter((r) => lines.includes(script[Number(r.dataset.index)]));
    player.ontimeupdate = () => {
      const t = player.currentTime;
      spoken.forEach((row, k) => row.classList.toggle('speaking', Boolean(data.lines[k]) && t >= data.lines[k].start && t < data.lines[k].end));
    };
    player.onended = () => spoken.forEach((row) => row.classList.remove('speaking'));
    player.play().catch(() => {});
  } catch (err) {
    status.textContent = '';
    note(err.message, true);
  }
}

$('#script-play').addEventListener('click', () => playScript(script));
