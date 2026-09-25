# Speech Model Toolkit

One web app for training your own speech models locally, with a tab for each:

- **Wake Word**: type a phrase, then train an [openWakeWord](https://github.com/dscripka/openWakeWord) model (`.onnx` + `.tflite`) and test it live with your microphone.
- **Voice**: record yourself (or upload a dataset), then train a [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voice for Home Assistant.
- **Test Lab**: say your wake word and hear your voice answer, optionally with an AI model answering your questions.
- **Settings**: AI connections, speech recognition and audio devices.

No coding required. Everything runs in one Docker container on your machine; the only thing that ever leaves it is what you send to an AI provider you configure yourself.

This project combines [easy-wakeword-trainer](https://github.com/mylegitches/easy-wakeword-trainer) and [piper-voice-helper](https://github.com/mylegitches/piper-voice-helper).

---

## Quick start

Requirements: Docker, ~40 GB of disk (image + training data), and ideally an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone https://github.com/mylegitches/speech-model-toolkit.git
cd speech-model-toolkit
docker compose up --build
```

Open **http://localhost:8765**. The Home tab shows the status of both tools; switch between them with the tabs at the top. A tool keeps running when you switch tabs, so you can watch one train while you use the other.

No GPU? Use `docker compose -f docker-compose.cpu.yml up --build`. Wake word training is fine on a CPU (~7–10 min per model). Voice training on a CPU takes days.

### Microphone access

Browsers only allow the microphone on `localhost` or HTTPS. To record from another machine, either tunnel (`ssh -L 8765:localhost:8765 <server>`, then open http://localhost:8765) or, in Chrome, add `http://<server-ip>:8765` to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

The port is published on all interfaces so other machines on your network can reach it. There is no login, so to keep it local only, change the port mapping in `docker-compose.yml` to `"127.0.0.1:8765:8765"`.

### On your phone

The whole app works on phones: on small screens the tabs move to a bottom bar, and buttons and fields are sized for touch. For the microphone the page must be served over **HTTPS** (e.g. behind a reverse proxy), which also lets you **add it to your home screen**: the Home tab offers this, and it then opens full screen like an app. While the Test Lab or the wake word tester is listening, the screen stays on so the phone doesn't sleep and cut the microphone.

Behind a reverse proxy, enable **WebSocket** support (the Test Lab and wake word tester stream audio over WebSockets) and turn off response buffering (live training logs). For nginx: `proxy_http_version 1.1`, `proxy_set_header Upgrade $http_upgrade`, `proxy_set_header Connection "upgrade"`, `proxy_buffering off`, `proxy_read_timeout 3600s`.

---

## Wake Word tab

1. **Download data** (first run only). The tab shows which training assets are missing and downloads them with one click (~5–20 GB):

   | Asset | Size | Purpose |
   |---|---|---|
   | Piper TTS checkpoint (`en_US-libritts_r-medium.pt`) | ~1.8 GB | Generates positive training clips |
   | MIT room impulse responses (`mit_rirs/`) | small | Acoustic augmentation |
   | FMA background music (`fma/`) | ~400 MB | Background noise augmentation |
   | ACAV negative features (`features_neg.npy`) | ~4–17 GB | Negative training examples |
   | Validation features (`validation_set_features.npy`) | ~180 MB | Early stopping validation |

   You can also run it from the command line: `docker compose run --rm toolkit python scripts/prepare_wakeword_data.py` (add `--skip-acav` if you'll provide `features_neg.npy` yourself).
2. **Wake phrase**: type a phrase (letters, numbers and spaces, max 6 words). **Preview** plays it in one of the synthetic voices used for training.
3. **Train**: Generate → Augment → Train → Export, with a live log. Then download `.zip`, `.onnx` or `.tflite`.
4. **Test**: pick a model and a microphone, click **Start Listening** and say the phrase. The ring lights up on detection. Audio streams to the container as 16 kHz PCM over a WebSocket and openWakeWord scores every 80 ms frame.

Training settings (1000 samples, 10k steps, ...) are in `app/wakeword/config_template.yaml`.

## Voice tab

1. **Voice**: create a voice with a name, language and whether it should sound female or male (this picks a similar pretrained voice to start from).
2. **Record**, in one of two ways (they can be mixed):
   - **Read sentences**: read the prompts shown. Keys: `R` record/stop, `P` play back, `S` save & next, `K` skip.
   - **Speak freely**: talk naturally (a story, your day, a book read aloud; up to 30 minutes per take), or upload an audio or video file you already have (a voice memo, podcast, interview, movie; up to 8 GB). The take is transcribed locally with Whisper (the model chosen in Settings) and cut into sentence-sized clips at sentence ends, pauses and speaker turns. Review them before saving: ▶ to listen, fix any wrong words (the text must match exactly what was said), untick clips with mistakes, music or other voices.
     - **Background noise**: *Keep*, *Reduce* (steady hiss and hum; the default) or *Remove* (RNNoise, a speech denoiser that also removes music, traffic and crowds). It can be changed per take while reviewing; the original audio is kept.
     - **Several people talking** (ticked automatically for video files): every clip is also assigned to a speaker, using the same speaker-recognition model as Auto-detect. The review then starts with **Who do you want?**: one card per speaker with how much they talk, three ▶ samples and a quote. Pick one (or several, if one person was split into two groups) and only their clips are listed. If the voice already has 5+ recordings, the speaker who sounds most like them is marked and preselected. Only use recordings of people who agreed to have their voice cloned.

   50 recordings is the minimum and 300+ sounds much better. Already have a prepared dataset? Upload a zip (`metadata.csv` with `file|text` lines plus audio, or audio files with matching `.txt` transcripts).
3. **Train**: pick a preset:

   | Preset | Time | |
   |---|---|---|
   | Quick test | 30 min | rough, to check everything works |
   | **Good** (default) | 3 hours | |
   | Best | 8 hours | |
   | Until I stop it | no limit | press **Stop** when it sounds right |

   **Starting voice**: training fine-tunes a pretrained Piper voice, and one that already sounds like you gets there faster and cleaner. The Train card lists the pretrained voices for your language; press ▶ to hear each one (sample clips from [piper-samples](https://rhasspy.github.io/piper-samples/)), or pick **Auto-detect** (the default): when training starts it compares up to 24 of your recordings with every candidate using a speaker-recognition model ([SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), runs locally) and starts from the closest one. **Find match** shows the ranking with similarity scores before you train. It needs at least 5 recordings; with fewer, it falls back to a voice of your language and chosen gender.

   **Advanced settings** has the same choice for other languages plus a custom checkpoint or training from scratch, and hours, max epochs, batch size (auto-picked from GPU memory), device and sample rate. **Train more** continues where the last run stopped.
4. **Test & install**: type anything and click **Speak**. During training, **Export latest version now** lets you hear progress. Then click **Download for Home Assistant**:
   1. In Home Assistant open **Settings → Add-ons → Piper → Open Web UI** and upload the `.onnx` and `.onnx.json` files (or copy them to `/share/piper`).
   2. Restart the Piper add-on if the voice doesn't show up.
   3. In **Settings → Voice assistants**, pick your voice.

Voices are trained with piper1-gpl, the current Piper, which is what Home Assistant's Piper add-on (`wyoming-piper`) runs.

## Test Lab tab

1. **Setup**: pick one of your wake words and a reply voice: any trained voice (its newest export) or the default Piper voice (*en_US-lessac-medium*, downloaded once, ~60 MB).
2. **Listen**: click **Start listening** and say the wake word.
   - **No AI connection**: it replies *"Your voice wakeword model was trained and triggered successfully."* (editable in Settings).
   - **With an AI connection**: after the wake word, ask your question. It records until you stop talking, transcribes it locally with Whisper, asks the AI, and speaks the answer. The last few exchanges are kept as context; **New conversation** clears them.
3. **Conversation**: every reply can be replayed and, if allowed in Settings, downloaded as `.wav`. You can also type a question instead of speaking.

The microphone is ignored while a reply plays, so the reply can't trigger the wake word.

## Settings tab

- **AI connections**: save as many as you like and pick which one the Test Lab uses. Choose a provider (the base URL fills in), paste a key, click **Load models** to pick from the provider's live model list, and **Test connection** to check it end to end.

  | Provider | Notes |
  |---|---|
  | OpenAI, Anthropic, Google Gemini | native APIs |
  | OpenRouter, Groq, Mistral, DeepSeek, xAI, Together, Fireworks, Cerebras, Cohere, Perplexity | OpenAI-compatible APIs (Perplexity: type the model name) |
  | Ollama (local), Ollama Cloud | local needs no key; the container reaches it at `host.docker.internal:11434` |
  | LM Studio, Custom | any OpenAI-compatible server (vLLM, LocalAI, llama.cpp, LiteLLM, …) |

  Keys are stored in `data/settings.json` (readable only by its owner) and are never sent back to the browser. **There is no login**, so if you save keys, bind the port to this machine only (`"127.0.0.1:8765:8765"` in `docker-compose.yml`).
- **Assistant**: which connection answers, the system prompt (short, speakable answers by default), temperature, max tokens, and the fixed reply.
- **Speech recognition**: Whisper model size and language, silence before a question ends, and the longest question. Models download on first use (or with **Download model now**) and run on the CPU.
- **Audio**: microphone (with a level test), speaker (Chrome/Edge) with a test sound, and volume, all remembered by your browser and used in every tab. Also echo cancellation / noise suppression / auto gain for listening, wake word sensitivity (also used by the Wake Word tester), the pause after each reply, and whether generated `.wav` files can be downloaded.

---

## When something goes wrong

Every error is shown in plain words where it happened, with what to do about it (for example "the GPU ran out of memory: lower the batch size", "this file has no audio track", "the file is too large for your reverse proxy"). Unexpected errors say so and point to the server log.

The server log has one line per event (time, level, component): training and exports starting, finishing or failing (with timings), downloads, freeform takes, speaker detection, auto-detect results, Test Lab sessions and AI calls (provider, model, time; never keys or what was said):

```bash
docker compose logs -f toolkit
```

Set `LOG_LEVEL=DEBUG` under `environment:` in `docker-compose.yml` for every HTTP request too, or `WARNING` for problems only. Training runs also keep their full output in `data/voice/voices/<name>/training/train.log`.

Large uploads (movies) need a generous upload limit on your reverse proxy, e.g. nginx `client_max_body_size 8g;`. Uploads are stored under `data/tmp` while they arrive, so they don't fill the container's own disk.

---

## Where things are stored

Everything is in `./data`, bind-mounted to `/data`:

```
data/
  settings.json      Settings tab (AI connections incl. keys, assistant, audio)
  wakeword/          wake word training assets + openWakeWord base models
  wakeword-models/   your wake words: <name>/<name>.onnx, .tflite, .zip, .yaml
  voice/
    voices/          one folder per voice: recordings, freeform takes awaiting review, training runs, exports
    checkpoints/     downloaded pretrained base voices
    speaker-match/   speaker recognition model + sample clips (Auto-detect, speaker detection)
    default-voices/  the Test Lab's default Piper voice
  stt-models/        Whisper models for the Test Lab
```

### Coming from the separate apps?

Move your existing data in and nothing needs to be downloaded again:

| Old location | New location |
|---|---|
| easy-wakeword-trainer `./data/*` | `./data/wakeword/` |
| easy-wakeword-trainer `./outputs/*` | `./data/wakeword-models/` |
| piper-voice-helper `./data/*` | `./data/voice/` |

---

## How it works

```
Browser ── /            landing page + tabs (app/static)
        ├─ /wakeword/   wake word UI + API (app/wakeword)
        ├─ /voice/      voice UI + API     (app/voice)
        ├─ /lab/        Test Lab UI + API  (app/lab)
        └─ /settings/   Settings UI + API  (app/settings)
                 │
         one FastAPI server (uvicorn, port 8765)
                 │
     ┌───────────┴─────────────┐
 system Python              /opt/piper-venv
 openWakeWord train.py,     piper1-gpl training
 piper-sample-generator,    and export (subprocess,
 onnx2tf (subprocesses)     $PIPER_PYTHON)
```

- The two trainers need incompatible PyTorch versions (openWakeWord's recipe pins torch 2.2, piper1-gpl needs a newer one), so the image has two Python environments. The web server and wake word training use the system Python; voice training runs as a subprocess in `/opt/piper-venv`.
- Each tool is its own FastAPI sub-app mounted under its prefix, and its page uses relative URLs. The landing page shows each tool in an iframe that is created on first visit and then kept alive.
- All three pages share `app/static/theme.css` (colors, cards, buttons, inputs, progress, logs). Tool stylesheets only add their own widgets.
- `GET /api/status` combines the tools' status for the Home tab and the busy dots on the tabs.
- The Test Lab streams 16 kHz microphone audio over one WebSocket (`/lab/api/session`). The server runs openWakeWord, then Silero VAD to find the end of your question, faster-whisper to transcribe it, the AI provider (plain HTTP, no vendor SDKs; see `app/settings/providers.py`), and Piper for the spoken reply.
- Tests: `pip install pytest httpx && pytest` (provider request shapes and settings key masking).
- Each tool runs one training job at a time (a second request gets HTTP 409). The two tools can train at the same time, but they share the GPU, and the Home tab warns you when both are running.

### Known quirks (inherited, handled)

- **Wake word validation OOM**: the full validation set (~481k rows) silently OOM-kills training at step 7500, so a 50k-row copy is created automatically and used instead.
- **Wake word `onnx_tf` error**: openWakeWord's `train.py` exits non-zero after writing the ONNX because `onnx_tf` isn't installed. That's expected; `onnx2tf` does the TFLite conversion.
- **Wake word DataLoader**: `num_workers` is patched to 0 at build time to avoid the >2 GB IPC pipe limit.
- **Piper**: the MOS checkpoint callback is disabled so training works offline, and ONNX export uses the TorchScript exporter because the dynamo exporter (default since torch 2.9) fails on Piper models.

---

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0): wake word training
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (MIT): synthetic wake word clips
- [onnx2tf](https://github.com/PINTO0309/onnx2tf) (MIT): TFLite export
- [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0): voice training and synthesis, installed at build time
- [Piper Recording Studio](https://github.com/rhasspy/piper-recording-studio) by Michael Hansen (MIT): prompts, recorder approach and dataset export (`export_dataset/`, `prompts/`; see `LICENSE-piper-recording-studio.md`)
- [TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy) by Erik Bjorgan (MIT): pretrained checkpoint lists and training workflow (`app/voice/checkpoints/`)
