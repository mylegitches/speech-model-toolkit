/* app.js — easy-wakeword-trainer frontend
 *
 * No framework. Relative API URLs so Electron can load http://127.0.0.1:8000 later.
 * No LLM or cloud-AI calls — all communication is to the local FastAPI server.
 *
 * Flow:
 *   1. On load → GET /api/health
 *      - If data ready  → show form, enable Train button
 *      - If data missing → show prepare section with "Download Data" button
 *   2. "Download Data" → POST /api/prepare → SSE /api/prepare/events
 *      - Streams live download log
 *      - On "done" → re-check health → show form
 *   3. "Train" → POST /api/train → SSE /api/train/{id}/events
 *      - Streams live training log
 *      - On "done" → show download buttons
 */

const STAGE_ORDER = ["generate", "augment", "train", "export"];

// ── DOM refs ──────────────────────────────────────────────────────────────
const phraseInput     = document.getElementById("phrase-input");
const trainBtn        = document.getElementById("train-btn");
const previewBtn      = document.getElementById("preview-btn");
const previewAudio    = document.getElementById("preview-audio");
const prepareSection  = document.getElementById("prepare-section");
const prepareTitle    = document.getElementById("prepare-title");
const prepareSubtitle = document.getElementById("prepare-subtitle");
const prepareBtn      = document.getElementById("prepare-btn");
const missingList     = document.getElementById("missing-list");
const prepareLog      = document.getElementById("prepare-log");
const progressSection = document.getElementById("progress-section");
const progressLabel   = document.getElementById("progress-label");
const logPanel        = document.getElementById("log-panel");
const downloadSection = document.getElementById("download-section");
const errorSection    = document.getElementById("error-section");
const errorMsg        = document.getElementById("error-msg");
const modelNameOut    = document.getElementById("model-name-out");
const dlZip           = document.getElementById("dl-zip");
const dlOnnx          = document.getElementById("dl-onnx");
const dlTflite        = document.getElementById("dl-tflite");

const stagePills = {
  generate: document.getElementById("stage-generate"),
  augment:  document.getElementById("stage-augment"),
  train:    document.getElementById("stage-train"),
  export:   document.getElementById("stage-export"),
};

// ── State ─────────────────────────────────────────────────────────────────
let dataReady      = false;
let jobRunning     = false;
let prepareRunning = false;

// ── Init ──────────────────────────────────────────────────────────────────
checkHealth();
phraseInput.addEventListener("input", updateTrainBtn);
trainBtn.addEventListener("click", () => startTraining(phraseInput.value.trim()));
phraseInput.addEventListener("keydown", (e) => { if (e.key === "Enter") trainBtn.click(); });
prepareBtn.addEventListener("click", startPrepare);

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch("api/health");
    const data = await res.json();
    dataReady = data.ready;

    if (!data.ready) {
      showPrepareSection(data.missing, data.assets);
    } else {
      hidePrepareSection();
    }
  } catch {
    dataReady = false;
    showPrepareSection(["Could not reach the server — is Docker running?"], {});
  }
  updateTrainBtn();
}

// ── Prepare section helpers ───────────────────────────────────────────────
function showPrepareSection(missing, assets) {
  prepareSection.classList.remove("hidden");
  missingList.innerHTML = "";
  missing.forEach((name) => {
    const li = document.createElement("li");
    const desc = assets[name] ? ` — ${assets[name].description}` : "";
    li.textContent = name + desc;
    missingList.appendChild(li);
  });
}

function hidePrepareSection() {
  prepareSection.classList.add("hidden");
}

function setPrepareState(running) {
  prepareRunning = running;
  prepareBtn.disabled = running;
  prepareBtn.textContent = running ? "Downloading…" : "Download Data";
}

// ── First-run data download ───────────────────────────────────────────────
async function startPrepare() {
  setPrepareState(true);
  missingList.innerHTML = "";
  prepareLog.textContent = "";
  prepareLog.classList.remove("hidden");
  prepareTitle.textContent = "Downloading training data…";
  prepareSubtitle.textContent =
    "This is a one-time download (~5–20 GB). Do not close the browser.";

  let jobId;
  try {
    const res  = await fetch("api/prepare", { method: "POST" });
    const data = await res.json();

    if (data.status === "already_ready") {
      onPrepareComplete();
      return;
    }
    jobId = data.job_id;
  } catch (e) {
    appendPrepareLog("ERROR: " + e.message);
    setPrepareState(false);
    return;
  }

  connectPrepareSSE(jobId);
}

function connectPrepareSSE(jobId) {
  const es = new EventSource("api/prepare/events");

  es.addEventListener("running", (e) => appendPrepareLog(e.data));
  es.addEventListener("log",     (e) => appendPrepareLog(e.data));

  es.addEventListener("done", () => {
    appendPrepareLog("✓ All assets ready.");
    es.close();
    onPrepareComplete();
  });

  // Server sent an explicit "error" event (training pipeline error)
  es.addEventListener("error", (e) => {
    if (e.data) {
      appendPrepareLog("ERROR: " + e.data);
      setPrepareState(false);
      es.close();
    }
    // if e.data is empty this is an SSE transport error, handled by es.onerror below
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    appendPrepareLog("[SSE connection dropped — switching to polling…]");
    es.close();
    pollPrepare();
  };
}

async function pollPrepare(intervalMs = 3000) {
  // Track how many lines we've already shown so we never re-append history.
  // Count current lines in the panel from before SSE dropped.
  let shownLines = prepareLog.textContent
    ? prepareLog.textContent.split("\n").length
    : 0;

  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const res  = await fetch("api/prepare/status");
      const data = await res.json();
      const lines = data.log_lines || [];
      // Only append lines we haven't shown yet
      lines.slice(shownLines).forEach(appendPrepareLog);
      shownLines = lines.length;
      if (data.stage === "done")  { onPrepareComplete(); return; }
      if (data.stage === "error") {
        appendPrepareLog("ERROR: " + (data.error || "Download failed."));
        setPrepareState(false);
        return;
      }
    } catch { /* ignore blips */ }
  }
}

function onPrepareComplete() {
  setPrepareState(false);
  prepareTitle.textContent  = "✓ Training data ready";
  prepareSubtitle.textContent = "";
  prepareBtn.classList.add("hidden");
  dataReady = true;
  updateTrainBtn();
  // Re-verify with the server
  checkHealth();
}

function appendPrepareLog(line) {
  if (!line) return;
  prepareLog.textContent += (prepareLog.textContent ? "\n" : "") + line;
  prepareLog.scrollTop = prepareLog.scrollHeight;
}

// ── Button state ──────────────────────────────────────────────────────────
function updateTrainBtn() {
  const hasText = phraseInput.value.trim().length > 0;
  trainBtn.disabled    = !hasText || !dataReady || jobRunning;
  previewBtn.disabled  = !hasText || jobRunning;
}

// ── Phrase preview (Piper TTS) ────────────────────────────────────────────
previewBtn.addEventListener("click", async () => {
  SMT.unlockAudio(previewAudio);  // phones: allow playing once the clip arrives
  const phrase = phraseInput.value.trim();
  if (!phrase) return;

  previewBtn.classList.add("loading");
  previewBtn.disabled = true;
  previewBtn.textContent = "⏳ Generating…";

  try {
    const res = await fetch("api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert("Preview failed: " + (err.detail || res.statusText));
      return;
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    previewAudio.src = url;
    await SMT.applyOutput(previewAudio);
    previewAudio.play();
    previewAudio.onended = () => URL.revokeObjectURL(url);
  } catch (e) {
    alert("Preview error: " + e.message);
  } finally {
    previewBtn.classList.remove("loading");
    previewBtn.textContent = "🔊 Preview";
    updateTrainBtn();
  }
});

// ── Phrase validation (mirrors server rules) ──────────────────────────────
function clientValidate(phrase) {
  if (!phrase) return "Phrase must not be empty.";
  if (!/^[a-z0-9 ]+$/.test(phrase.toLowerCase()))
    return "Phrase may only contain letters, numbers, and spaces.";
  if (phrase.trim().split(/\s+/).length > 6)
    return "Phrase must be at most 6 words.";
  return null;
}

// ── Training ──────────────────────────────────────────────────────────────
async function startTraining(phrase) {
  const err = clientValidate(phrase);
  if (err) { showError(err); return; }

  resetTrainUI();
  setJobRunning(true);

  let jobId;
  try {
    const res = await fetch("api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    jobId = data.job_id;
    progressLabel.textContent = `Training "${data.model_name}"…`;
    modelNameOut.textContent  = data.model_name;
  } catch (e) {
    showError(e.message);
    setJobRunning(false);
    return;
  }

  progressSection.classList.remove("hidden");
  connectTrainSSE(jobId);
}

function connectTrainSSE(jobId) {
  const es = new EventSource(`api/train/${jobId}/events`);

  es.addEventListener("log",  (e) => appendTrainLog(e.data));
  es.addEventListener("info", (e) => appendTrainLog(e.data));

  STAGE_ORDER.forEach((stage) => {
    es.addEventListener(stage, (e) => {
      activateStage(stage);
      appendTrainLog(e.data);
    });
  });

  es.addEventListener("done", () => {
    markAllStagesDone();
    progressLabel.textContent = "Done!";
    showDownloads(jobId);
    setJobRunning(false);
    es.close();
  });

  es.addEventListener("error", (e) => {
    showError(e.data || "Training failed. See log for details.");
    setJobRunning(false);
    es.close();
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    appendTrainLog("[SSE dropped — polling status…]");
    es.close();
    pollTrain(jobId);
  };
}

async function pollTrain(jobId, intervalMs = 3000) {
  while (true) {
    await new Promise((r) => setTimeout(r, intervalMs));
    try {
      const res  = await fetch(`api/train/${jobId}/status`);
      const data = await res.json();
      progressLabel.textContent = `Stage: ${data.stage}`;
      if (data.stage === "done")  { markAllStagesDone(); showDownloads(jobId); setJobRunning(false); return; }
      if (data.stage === "error") { showError(data.error || "Training failed."); setJobRunning(false); return; }
      if (STAGE_ORDER.includes(data.stage)) activateStage(data.stage);
    } catch { /* ignore */ }
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────
function appendTrainLog(line) {
  logPanel.textContent += (logPanel.textContent ? "\n" : "") + line;
  logPanel.scrollTop = logPanel.scrollHeight;
}

function activateStage(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  STAGE_ORDER.forEach((s, i) => {
    const pill = stagePills[s];
    if (!pill) return;
    pill.classList.remove("active", "done");
    if (i < idx)      pill.classList.add("done");
    else if (i === idx) pill.classList.add("active");
  });
}

function markAllStagesDone() {
  STAGE_ORDER.forEach((s) => {
    const pill = stagePills[s];
    if (pill) { pill.classList.remove("active"); pill.classList.add("done"); }
  });
}

function showDownloads(jobId) {
  dlZip.href    = `api/train/${jobId}/download`;
  dlOnnx.href   = `api/train/${jobId}/download/onnx`;
  dlTflite.href = `api/train/${jobId}/download/tflite`;
  downloadSection.classList.remove("hidden");
  // Refresh the tester model list so the new model appears immediately
  document.dispatchEvent(new Event("trainingDone"));
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorSection.classList.remove("hidden");
}

function resetTrainUI() {
  progressSection.classList.add("hidden");
  downloadSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  logPanel.textContent      = "";
  progressLabel.textContent = "Starting…";
  STAGE_ORDER.forEach((s) => {
    const pill = stagePills[s];
    if (pill) pill.classList.remove("active", "done");
  });
}

function setJobRunning(running) {
  jobRunning = running;
  phraseInput.disabled = running;
  updateTrainBtn();
}

// =============================================================================
// TESTER — browse models, stream mic audio, show live detection
// =============================================================================

const modelSelect    = document.getElementById("model-select");
const micSelect      = document.getElementById("mic-select");
const micActiveLabel = document.getElementById("mic-active-label");
const micBtn         = document.getElementById("mic-btn");
const detectorRing   = document.getElementById("detector-ring");
const detectorLbl    = document.getElementById("detector-label");
const scoreBar       = document.getElementById("score-bar");
const scoreVal       = document.getElementById("score-val");
const vuRow          = document.getElementById("vu-row");
const vuBar          = document.getElementById("vu-bar");
const vuVal          = document.getElementById("vu-val");

let testerWs       = null;
let audioCtx       = null;
let mediaStream    = null;
let scriptNode     = null;
let detectionTimer = null;

// Populate microphone dropdown
async function populateMicList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter(d => d.kind === "audioinput");
    const prev = micSelect.value;
    micSelect.innerHTML = '<option value="">🎙 Default microphone</option>';
    mics.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${micSelect.options.length}`;
      micSelect.appendChild(opt);
    });
    // Restore previous selection, else the microphone chosen in Settings
    const wanted = prev || SMT.get("input");
    if (wanted && [...micSelect.options].some(o => o.value === wanted)) micSelect.value = wanted;
  } catch (e) {
    console.warn("Could not enumerate audio devices:", e);
  }
}
populateMicList();
// Re-enumerate after user grants permission (labels appear after first grant)
navigator.mediaDevices.addEventListener("devicechange", populateMicList);

// Populate model dropdown on load
async function loadModels() {
  try {
    const res = await fetch("api/models");
    const { models } = await res.json();
    modelSelect.innerHTML = '<option value="">— select a model —</option>';
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name}  (${m.size_kb} KB)`;
      modelSelect.appendChild(opt);
    });
    micBtn.disabled = models.length === 0;
  } catch (e) {
    console.warn("Could not load models:", e);
  }
}
loadModels();
// Refresh model list after a training job completes
document.addEventListener("trainingDone", loadModels);

modelSelect.addEventListener("change", () => {
  if (testerWs) stopListening();
  micBtn.disabled = !modelSelect.value;
  micBtn.textContent = "🎤 Start Listening";
  micBtn.classList.remove("listening");
  resetDetector();
});

micBtn.addEventListener("click", () => {
  if (testerWs) {
    stopListening();
  } else {
    startListening();
  }
});

async function startListening() {
  const modelName = modelSelect.value;
  if (!modelName) return;
  // Phones only start audio from a tap: create the context before any await
  audioCtx = new AudioContext({ sampleRate: 16000 });
  SMT.keepAwake(true);

  const selectedDeviceId = micSelect.value;
  const audioPrefs = (await SMT.serverSettings())?.audio || {};
  const audioConstraints = {
    sampleRate: 16000,
    echoCancellation: audioPrefs.echoCancellation ?? true,
    noiseSuppression: audioPrefs.noiseSuppression ?? true,
    autoGainControl: audioPrefs.autoGainControl ?? true,
  };
  try {
    mediaStream = await SMT.openMic(audioConstraints, selectedDeviceId);
    // Re-populate now that we have permission (labels may have been blank)
    await populateMicList();
    // Show which mic is actually active
    const track = mediaStream.getAudioTracks()[0];
    const label = track?.label || selectedDeviceId || "Default microphone";
    micActiveLabel.textContent = `🎙 ${label}`;
    micActiveLabel.style.display = "block";
  } catch (e) {
    alert("Microphone access denied: " + e.message);
    stopListening();
    return;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  testerWs = new WebSocket(`${proto}://${location.host}${location.pathname.replace(/[^/]*$/, "")}api/test/${encodeURIComponent(modelName)}`);
  testerWs.binaryType = "arraybuffer";

  testerWs.onopen = async () => {
    micBtn.textContent = "⏹ Stop Listening";
    micBtn.classList.add("listening");
    detectorLbl.textContent = "Listening…";
    try {
      await startMicCapture();
      vuRow.style.display = "flex";
    } catch (e) {
      console.error("Mic capture failed:", e);
      detectorLbl.textContent = "Mic error: " + e.message;
      stopListening();
    }
  };

  testerWs.onmessage = (evt) => {
    const { score, detected } = JSON.parse(evt.data);
    updateDetector(score, detected);
  };

  testerWs.onerror = (e) => console.error("Tester WS error", e);
  testerWs.onclose = () => stopListening();
}

async function startMicCapture() {
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate: 16000 });
  if (audioCtx.state === "suspended") await audioCtx.resume();

  // AudioWorklet runs in a dedicated audio thread — no main-thread blocking,
  // no ScriptProcessorNode deprecation warning.
  await audioCtx.audioWorklet.addModule("../static/mic-processor.js");

  const source   = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode     = new AudioWorkletNode(audioCtx, "mic-processor");

  scriptNode.port.onmessage = (evt) => {
    // Compute RMS for VU meter before handing buffer to WebSocket
    const int16 = new Int16Array(evt.data);
    let sumSq = 0;
    for (let i = 0; i < int16.length; i++) {
      const s = int16[i] / 32768;
      sumSq += s * s;
    }
    const rms = Math.sqrt(sumSq / int16.length);
    updateVu(rms);

    if (!testerWs || testerWs.readyState !== WebSocket.OPEN) return;
    testerWs.send(evt.data); // ArrayBuffer — WS copies it, buffer stays valid
  };

  source.connect(scriptNode);
  // No connection to destination — we don't want mic audio playing back
}

function stopListening() {
  if (scriptNode)  { scriptNode.port.close(); scriptNode.disconnect(); scriptNode = null; }
  if (audioCtx)    { audioCtx.close();        audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (testerWs && testerWs.readyState < 2) testerWs.close();
  testerWs = null;
  SMT.keepAwake(false);

  micBtn.textContent = "🎤 Start Listening";
  micBtn.classList.remove("listening");
  micActiveLabel.style.display = "none";
  vuRow.style.display = "none";
  updateVu(0);
  detectorLbl.textContent = "—";
  resetDetector();
}

function updateVu(rms) {
  // rms is 0..1 (float), map to display percentage with some headroom
  const pct = Math.min(100, rms * 400); // amplify so normal speech ~50-80%
  vuBar.style.width = pct + "%";
  vuBar.classList.toggle("hot",  pct > 60);
  vuBar.classList.toggle("clip", pct > 90);
  vuVal.textContent = rms > 0 ? rms.toFixed(3) : "—";
}

function updateDetector(score, detected) {
  const pct = Math.round(score * 100);
  scoreBar.style.width = pct + "%";
  scoreVal.textContent = `score: ${score.toFixed(3)}`;

  if (detected) {
    detectorRing.classList.add("active");
    scoreBar.classList.add("hot");
    detectorLbl.textContent = "✅ Detected!";

    clearTimeout(detectionTimer);
    detectionTimer = setTimeout(() => {
      detectorRing.classList.remove("active");
      scoreBar.classList.remove("hot");
      detectorLbl.textContent = "Listening…";
    }, 1500);
  } else {
    if (!detectorRing.classList.contains("active")) {
      scoreBar.classList.remove("hot");
    }
  }
}

function resetDetector() {
  clearTimeout(detectionTimer);
  detectorRing.classList.remove("active");
  scoreBar.classList.remove("hot");
  scoreBar.style.width = "0%";
  scoreVal.textContent = "";
}
