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
  options = await (await fetch('api/options')).json();

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
$('#voice-select').addEventListener('change', (e) => {
  SMT_lab('voice', e.target.value);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'voice', voice: e.target.value }));
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
    note(`Microphone error: ${err.message}`, true);
    stop();
    return;
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const base = location.pathname.replace(/[^/]*$/, '');
  ws = new WebSocket(`${proto}://${location.host}${base}api/session`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = async () => {
    ws.send(JSON.stringify({ type: 'start', wakeword, voice: $('#voice-select').value }));
    try {
      await startCapture();
    } catch (err) {
      note(`Microphone error: ${err.message}`, true);
      stop();
    }
  };
  ws.onmessage = (evt) => onMessage(JSON.parse(evt.data));
  ws.onclose = () => { if (ws) stop(); };

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
  button.textContent = 'Thinking…';
  try {
    const res = await fetch('api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: $('#voice-select').value, history: priorHistory }),
    });
    if (!res.ok) throw new Error(await res.text());
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
