# Speech Model Toolkit: Full Feature Writeup

> Source material for promo videos, user guides, posts and portfolio pages.
> Everything below describes what the software actually does today; numbers
> come from tests run on the real application (see [Verified results](#verified-results)).

---

## In one sentence

**Speech Model Toolkit is a self-hosted web app that lets anyone train their own custom wake word and clone a voice for text-to-speech, then try both together in a voice assistant, all running locally in one Docker container with no coding.**

## Short pitches

- **10 words:** Train your own wake word and voice. Locally. No code.
- **Tagline options:**
  - "Your voice. Your wake word. Your machine."
  - "Build a voice assistant that sounds like you, without sending a word to the cloud."
  - "From a phrase and a few minutes of speech to a working voice assistant."
- **Elevator pitch (30 s):** Smart home voice assistants make you use their wake words and their voices. Speech Model Toolkit lets you train both yourself. Type a phrase and it trains a wake word detector. Read some sentences, talk freely, or import a video, and it fine-tunes a text-to-speech voice that sounds like you, automatically picking the best starting voice. Then the Test Lab turns them into a working assistant: say your wake word, ask a question, and hear the answer in your own voice, optionally powered by any AI model you choose. Everything runs in one Docker container on your own hardware, and the results drop straight into Home Assistant.

---

## The problem it solves

- Custom wake words and custom TTS voices exist in the open-source world (openWakeWord, Piper), but training them means notebooks, command lines, dataset formatting, dependency conflicts and GPU wrangling.
- Creating a voice dataset by hand is tedious: every clip needs an exact transcript and consistent audio.
- Nothing let you *try* a custom wake word and a custom voice together as an assistant before deploying them.
- Cloud voice services require sending your voice and conversations to someone else's servers.

**Speech Model Toolkit turns all of that into a guided web interface**, with privacy by default: training, transcription, speaker recognition and speech synthesis all run locally. The only thing that ever leaves the machine is what you choose to send to an AI provider you configure yourself.

## Who it's for

- **Home Assistant users** who want a personal wake word ("Hey Jarvis", "Yo Hal") and an assistant voice that sounds like them or a family member.
- **Makers and tinkerers** building local voice assistants on their own hardware.
- **Privacy-conscious users** who don't want cloud voice services.
- **Developers and ML hobbyists** who want a working, well-structured reference for openWakeWord + Piper training pipelines.

---

## Feature tour

The app is one page with five tabs: **Home, Wake Word, Voice, Test Lab, Settings**. On phones the tabs become an app-style bottom bar, and the whole thing can be installed to the home screen.

### 🏠 Home

- Live status of everything at a glance: whether the wake word training data is downloaded, how many wake words and voices exist, which GPU is available, what is training right now, and whether an AI is connected.
- Busy indicators (pulsing dots) on the tabs while something is training.
- A warning when both trainers run at once and share the GPU.
- An "Add to Home Screen" prompt on phones.

### 🎙️ Wake Word: train a custom wake word from just text

1. **One-click training data download** (~5–20 GB, one time): Piper TTS checkpoint, room impulse responses, background music, negative speech features and validation features, with a live log and automatic resume if the connection drops.
2. **Type a phrase** (e.g. "hey computer") and press **Preview** to hear how the synthetic voices will say it.
3. **Train**: the pipeline generates 1,000 synthetic spoken samples of the phrase with Piper, augments them with background noise and room echo, trains an openWakeWord detector, and exports **ONNX and TFLite** models. Four live stages (Generate → Augment → Train → Export) with a streaming log. Roughly 7–10 minutes on a CPU.
4. **Test it live**: pick the model and a microphone, press Start Listening, and say the phrase. A ring lights up on detection, with a live level meter and confidence score. Audio streams from the browser to the server over WebSocket and is scored every 80 ms.
5. **Download** as `.zip`, `.onnx` or `.tflite` for Home Assistant, ESPHome-style devices, or your own projects.

### 🗣️ Voice: clone a voice for Piper text-to-speech

**Step 1: Voice.** Create a voice with a name, language (59 recording-prompt languages) and whether it should sound female or male.

**Step 2: Dataset.** Build the training data, short clips with exact transcripts, in any mix of three ways:

- **Read sentences:** a teleprompter-style recorder with over 1,000 varied prompts per language (e.g. 1,150 for US English). Keyboard shortcuts (R record, P play, S save, K skip), a level meter with clipping warning, and a readiness bar (50 clips minimum, 300+ recommended, 1,000 ideal).
- **Speak freely:** just talk (a story, your day, a book read aloud) for up to 30 minutes. The take is transcribed locally with Whisper using word-level timestamps and **automatically cut into sentence-sized clips** at sentence ends, pauses and speaker turns.
- **Import file:** upload audio *or video* up to 8 GB each (MP4, MKV, AVI, MOV, WMV, MP3, WAV, FLAC and many more); only the sound is used. Import **several files, a whole folder with its subfolders, or drag and drop**, then tick the files you want from the list. Files are queued and processed one at a time.
  - **Multiple audio tracks** (movie languages, commentary): you choose the track, and the one in the voice's language is preselected.
  - **Surround sound (5.1/7.1):** only the **center channel** is used by default. That's where film dialogue is mixed, away from the music and effects.
  - **Speaker detection (diarization):** for interviews, podcasts and movies, every clip is assigned to a speaker. A **"Who do you want?"** screen shows one card per speaker with talk time, three playable samples and a quote. Pick one and only their clips are kept. If the voice already has recordings, the speaker who sounds most like them is **marked and preselected**.
  - **People recognised across files:** for a series of episodes with recurring people, every speaker is matched against the people found in earlier files by voice fingerprint, so the same person keeps the same card and name everywhere ("also in 3 other files"). Click a name to rename, merge duplicates (likely duplicates are suggested, and several can be merged at once), search long cast lists, and mark **This is the voice** to preselect that person in every file and **save their clips from all files at once**.
  - **AI speaker identification:** import a whole series (every season, every episode) and the AI names the characters. It reads a few transcribed lines per person, gets the cast from TVmaze (via an IMDb ID in the file names or the show's folder name) and can search the web (OpenRouter, Perplexity, Gemini, Anthropic, OpenAI). Cards show *AI: Tony Soprano (James Gandolfini) · high*; "Person N" cards can be named automatically, and cards the AI says are the same character are merged when their voices agree, so the series ends up as **one card per character**.
  - **Prepared datasets** (LJSpeech-style `metadata.csv` or audio + `.txt` pairs) can be imported as a zip.
- **Review before anything is added:** every automatically cut clip can be played, its transcript corrected, and unticked if it has mistakes, music or other voices.
- **Background noise removal** per take, switchable while reviewing (the original audio is always kept):
  - *Keep as is*
  - *Reduce*: steady hiss and hum (high-pass + FFT denoiser)
  - *Remove*: RNNoise neural speech denoiser, which also removes music, traffic and crowds

**Step 3: Train.**

- **Starting voice with ▶ samples:** training fine-tunes a pretrained Piper voice. The app lists every pretrained voice for the language (11 for US English), each with a play button for its official sample.
- **Auto-detect (the default):** compares up to 24 of your clips with every candidate using a speaker-recognition neural network (ECAPA-TDNN) and starts from the closest-sounding one. **Find match** shows the ranking with similarity scores before you train.
- **Presets:** Quick test (30 min), Good (3 h), Best (8 h), or Until I stop it. Advanced settings cover epochs, batch size (auto-picked from GPU memory), device, sample rate, other languages, custom checkpoints or training from scratch. **Train more** continues where the last run stopped.
- Live progress: stages, a time bar and the full training log.

**Step 4: Test & install.**

- Type anything and hear the voice speak it, even *during* training ("Export latest version now").
- **Download for Home Assistant:** `.onnx` + `.onnx.json`, named the way Home Assistant's Piper add-on expects, with step-by-step install instructions on the page.

### 🧪 Test Lab: your wake word + your voice = a working assistant

- Pick a trained wake word and a reply voice (any trained voice or a built-in default).
- Press **Start listening** and say the wake word:
  - **Without AI:** it answers *"Your voice wakeword model was trained and triggered successfully."* in the chosen voice. That's a one-click end-to-end check of both models.
  - **With an AI connection:** after the wake word, ask anything. It detects when you stop talking (Silero VAD), transcribes you locally (Whisper), asks the AI, and **speaks the answer in your trained voice**. Conversation context is kept across turns.
- Live state display (Listening → Heard it! → Your turn → Transcribing → Thinking → Speaking), detector ring, confidence bar and level meter.
- Chat-style transcript; every reply can be replayed or downloaded as `.wav`.
- Type questions instead of speaking.
- The microphone is ignored while a reply plays, so the assistant can't trigger itself.

### ⚙️ Settings

- **AI connections:** save any number and choose which one the Test Lab uses. **17 providers:** OpenAI, Anthropic, Google Gemini, OpenRouter, Ollama (local), Ollama Cloud, Groq, Mistral AI, DeepSeek, xAI (Grok), Together AI, Fireworks AI, Cerebras, Cohere, Perplexity, LM Studio, and any custom OpenAI-compatible server (vLLM, LocalAI, llama.cpp, LiteLLM…).
  - The base URL fills in per provider, with a "Get a key" link.
  - **Load models** pulls the provider's live model list.
  - **Test connection** checks the key and sends a real test prompt, reporting latency.
  - API keys are stored only on your server and **never sent back to the browser** (masked as `sk-…abcd`).
- **Assistant:** system prompt (tuned for short, speakable answers), temperature, max tokens, and the fixed reply text.
- **Speaker identification (AI):** on/off, TVmaze cast lookup, web search, automatic naming and automatic merging.
- **Speech recognition:** Whisper model size (tiny → medium, English-only or multilingual), language, end-of-speech silence, maximum question length, and download status.
- **Audio:**
  - microphone with live level test
  - speaker output (Chrome/Edge) with test chime
  - volume
  - echo cancellation / noise suppression / auto gain
  - wake word sensitivity and pause after each reply
  - whether generated `.wav` files may be downloaded

  Device choices are remembered per browser and apply in every tab.

### 📱 Mobile and install

- Responsive layout with an app-style **bottom tab bar** on phones and touch-sized controls.
- **Installable** (web app manifest + icons): runs full screen from the home screen.
- iPhone-specific audio handling so replies and previews play reliably.
- **Keeps the screen awake** while listening so the phone doesn't cut the microphone.

### 🛟 Reliability: clear errors and logging everywhere

- Every failure is shown in plain English with what to do about it:
  - *"The GPU ran out of memory. Lower the batch size…"*
  - *"This file has no audio track."*
  - *"The file is too large for your reverse proxy…"*
  - *"Microphone access is blocked. Allow it in the browser…"*
- Failed training steps are diagnosed from their output: GPU or RAM out of memory, disk full, network failure, missing component, or permissions.
- Structured server logs (time, level, component) for every job: training, exports, downloads, takes, speaker detection, auto-detect, Test Lab sessions and AI calls, with timings. API keys and what people say never appear in the logs. Adjustable with `LOG_LEVEL`.
- Downloads resume after network stalls. Long jobs keep running when you switch tabs or close the browser.

---

## Verified results

Measured on the real application in its Docker container (CPU only):

| What | Result |
|---|---|
| Auto-detect starting voice | Recordings made with the Piper *lessac* voice matched **lessac at 95%** similarity; the next closest was 27%. First run 54 s (includes model download), ~14 s after. |
| Speaker detection | A two-person dialogue "movie" (MP4, quick turns, background noise): **10/10 clips assigned to the right speaker**. The target speaker was preselected at 94% similarity vs 22% for the other. |
| Freeform transcription | A 25 s noisy take became **6 clips with word-for-word correct transcripts** in 8 s. |
| Noise removal | Speech-to-noise ratio of a noisy recording: 16.6 dB originally → **36.4 dB with *Reduce*** → **52.6 dB with *Remove***, with speech level unchanged. |
| Test Lab end to end | Spoken wake word detected (score 0.94) → question transcribed exactly ("What is the capital of France?") → AI answered → reply spoken in the chosen voice. |
| Automated tests | 32 unit tests: AI provider request/response formats for all API styles, settings key masking, clip splitting, track selection, error explanations. |

---

## How it works (architecture)

```
Browser (desktop or phone, installable)
  └─ one page with 5 tabs; each tool is its own page in a kept-alive frame,
     so training logs keep streaming while you use another tab
        │  HTTPS + WebSockets (live audio) + server-sent events (live logs)
        ▼
One Docker container · FastAPI server (Python)
  ├─ /wakeword  openWakeWord pipeline: Piper sample generation → augmentation →
  │             DNN training → ONNX/TFLite export; live WebSocket tester
  ├─ /voice     dataset tools (prompt recorder, freeform + Whisper, video import,
  │             speaker detection, noise removal), Piper fine-tuning, export
  ├─ /lab       wake word → Silero VAD → Whisper → AI provider → Piper TTS
  ├─ /settings  AI connections (17 providers), speech, audio
  └─ two Python environments side by side:
       system Python   web app + openWakeWord training (PyTorch 2.2)
       /opt/piper-venv piper1-gpl voice training (newer PyTorch), run as a subprocess
```

### Technology

- **Backend:** Python, FastAPI, asyncio, WebSockets, server-sent events.
- **Speech ML:**
  - [openWakeWord](https://github.com/dscripka/openWakeWord) (wake word)
  - [Piper / piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (TTS training and synthesis)
  - [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (speech recognition)
  - [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (speaker embeddings for auto-detect and diarization)
  - [Silero VAD](https://github.com/snakers4/silero-vad) (voice activity detection)
  - RNNoise (neural denoising via ffmpeg)
  - onnx2tf (TFLite export)
- **Audio/video:** ffmpeg and ffprobe (decoding any format, track selection, center-channel extraction, filters).
- **AI:** direct HTTP clients for four API styles (OpenAI-compatible, Anthropic Messages, Google Gemini, Ollama), with no vendor SDKs.
- **Frontend:** framework-free HTML/CSS/JavaScript, one shared design system, AudioWorklet microphone capture, Screen Wake Lock, web app manifest.
- **Packaging:** a single Docker image with GPU (NVIDIA) and CPU compose files; everything persistent lives in one `./data` folder.

### Engineering highlights (good material for technical posts)

1. **Two incompatible ML stacks in one container.** openWakeWord's training recipe pins PyTorch 2.2 / NumPy 1.26; Piper's current trainer needs a much newer PyTorch. Instead of two containers, the image has two Python environments, and the web server drives Piper training as a subprocess.
2. **Merging two apps into one product.** Two separate projects (a wake word trainer and a voice trainer) became mounted sub-applications behind one server, with prefix-relative URLs, one shared theme, and iframes that stay alive, so a 3-hour voice training keeps streaming while you test a wake word.
3. **Auto-detect via speaker embeddings.** Each pretrained voice's official sample and the user's clips are embedded with ECAPA-TDNN. Cosine similarity ranks the candidates, and fine-tuning starts from the closest voice.
4. **Diarization without paid models or tokens, and across files.** Clips are cut at pauses, sentence ends and Whisper segment boundaries (usually speaker turns), embedded, grouped by average-linkage clustering on cosine distance, and matched against the voice's existing recordings to suggest the right person. Each speaker's mean fingerprint is also matched greedily (best pairs first, one person per speaker per file) against a per-voice library of people, refining that person's fingerprint as more of their speech is seen, so recurring people get stable identities across a whole series of files.
5. **Dialogue extraction from movies.** ffprobe lists the audio tracks; the voice's language is preselected (commentary tracks avoided). For 5.1/7.1 mixes, ffmpeg keeps only the center channel (`pan=mono|c0=FC`), where film dialogue lives.
6. **Noise removal, measured rather than assumed.** The first filter setting barely worked (<2 dB). Measuring noise and speech levels separately showed a 60 Hz hum was hiding the noise floor from the FFT denoiser. Removing it first, and running RNNoise at the 48 kHz it expects, gave +20 dB and +36 dB improvements.
7. **A real-time voice loop in the browser.** AudioWorklet captures 16 kHz PCM → WebSocket → wake word scoring every 80 ms → VAD end-of-speech → Whisper → LLM → Piper → playback, with the microphone muted during playback to prevent self-triggering.
8. **Mobile audio quirks handled.** On iOS, AudioContexts are created and audio elements unlocked inside the tap, because sound that arrives later (AI replies) would otherwise be silently blocked. Wake Lock keeps listening sessions alive.
9. **Security by default.** API keys stay server-side (masked, `chmod 600`). Paths are stripped from error messages. Uploads stream to disk (8 GB files never sit in memory). A `no-cache` revalidation policy stops updated containers from mixing new pages with stale cached scripts.
10. **AI identifies the characters of a series.** Voice fingerprints alone split one character into several cards (shouting vs. calm, a cold) and can't tell you names. The toolkit sends each person's transcribed lines, the file names and the real cast list (TVmaze, found via the IMDb ID or show folder) to any of 17 LLM providers, with provider-native web search where available (Anthropic's web search tool, Gemini's Google Search grounding, OpenRouter `:online`, Perplexity). Its answers name the cards and drive merging, but a merge also needs the voices to agree, so a wrong guess becomes a suggestion rather than a silent mistake.
11. **Operational polish.** Resumable downloads, structured logs, plain-English error explanations for failed steps, and a test suite for the parts that talk to external APIs.

---

## Typical user journeys (great for demo videos)

**1. "My own wake word in 10 minutes"**
Download the training data once → type "hey jarvis" → Preview → Train → watch the four stages → Test: say "hey jarvis" and the ring lights up → download for Home Assistant.

**2. "Clone my voice without reading a script"**
New voice → Dataset → Speak freely → talk about your weekend for 10 minutes → review the auto-cut clips → Train with Auto-detect (it shows the closest-sounding starting voice) → hear your voice read any text.

**3. "Pull one actor's voice out of an interview video"** (with permission)
Import file → choose the English 5.1 track (dialogue channel on) → speaker detection → "Who do you want?" → play the samples → pick the speaker → review → train.

**3b. "A whole series"** (with permission)
📁 Choose folder → the season's episodes are listed → Import → each episode is processed in turn → the same people are recognised in every episode → name the one you want and mark "This is the voice" → review → "Save Jerry's clips from all 10 files".

**4. "A talking assistant in my own voice"**
Settings → add an AI connection (e.g. OpenRouter or a local Ollama) → Test Lab → say your wake word → "What's the capital of France?" → hear the answer in your cloned voice.

**5. "From my phone"**
Open the app over HTTPS, choose Add to Home Screen, then record dataset sentences on the couch with the bottom tab bar.

### Suggested promo video storyboard (60–90 s)

1. Hook (0–5 s): *"What if your voice assistant had **your** wake word… and **your** voice?"*
2. Home tab overview: one app, five tabs, one Docker container (5–10 s).
3. Wake Word: type a phrase → Preview → Train stages → live test ring lighting up (15 s).
4. Voice: Speak freely → auto-cut clips → Auto-detect ranking → Train (15 s).
5. The movie trick: import video → track choice → "Who do you want?" speaker cards (10 s).
6. Test Lab payoff: say the wake word, ask a question, the answer comes back in the cloned voice (15 s).
7. Phone: installed app, bottom tabs (5 s).
8. Close: *"Local. Private. Open source. Speech Model Toolkit."* + repo link.

### Screens worth capturing

- Home tab with live status cards
- Wake Word training stages and the detector ring lighting up
- Dataset step with the three modes (Read sentences / Speak freely / Import file)
- Freeform review list with editable transcripts
- The audio track chooser and the "Who do you want?" speaker cards
- Starting voice list with similarity percentages and ▶ buttons
- Test Lab mid-conversation (state pill "Speaking", chat bubbles)
- Settings → AI connections with a green "✓ Connected" test result
- The phone layout with the bottom tab bar

### Social post ideas

- *"I merged my wake word trainer and my voice cloning tool into one self-hosted app, then added a Test Lab where your custom wake word triggers an AI that answers in **your** voice. Everything runs locally in one Docker container."*
- *"Fun engineering problem: two ML stacks that need incompatible PyTorch versions, one container. Solution: two Python environments, one server, subprocess orchestration."*
- *"Pulling clean dialogue out of movies for voice training: pick the English track automatically, keep only the 5.1 center channel, cluster the voices, and let the user pick the speaker from sample cards."*
- *"My first noise filter only removed 2 dB. Measuring properly revealed a 60 Hz hum was fooling the denoiser. Fixed: +36 dB."*
- *"Auto-detect picks the pretrained voice closest to yours: 95% match vs 27% for the runner-up, using speaker embeddings."*

---

## Requirements and deployment

- Docker (Linux, Windows or macOS host). An NVIDIA GPU with the NVIDIA Container Toolkit is strongly recommended for voice training; a CPU-only compose file is included (wake words train fine on a CPU).
- About 40 GB of disk: a ~25 GB image plus training data.
- `docker compose up -d --build`, then open `http://localhost:8765`.
- For phones and remote access: serve it over HTTPS (e.g. Nginx Proxy Manager with WebSockets enabled and a large upload limit).

## Honest limitations

- Voice training on a CPU is very slow (days for a good voice); a GPU is recommended.
- AI speaker identification is only as good as the model and the lines it sees; well-known shows work best, and the user's own names always win.
- Speaker detection doesn't separate people talking over each other. Those clips should be unticked during review.
- Auto-detect scores come from each pretrained voice's short official sample, so treat the ranking as a strong hint and listen to the top candidates.
- The wake word pipeline generates English-pronounced samples (Piper's multi-speaker English model).
- There is no built-in login. It's meant for a home network or behind a reverse proxy.

## Ethics

The app is built for cloning **your own** voice or the voice of someone who has agreed to it. The import screen says so explicitly. Voice cloning of real people without consent can be illegal and harmful.

## Credits and licenses

Built on outstanding open-source work:

- openWakeWord (Apache-2.0)
- piper-sample-generator (MIT)
- onnx2tf (MIT)
- piper1-gpl (GPL-3.0, installed at build time)
- Piper Recording Studio (MIT: prompts and dataset export)
- TextyMcSpeechy (MIT: pretrained checkpoint lists)
- faster-whisper (MIT)
- SpeechBrain (Apache-2.0)
- Silero VAD (MIT)
- RNNoise models by GregorR (downloaded at build time)
- ffmpeg

Combines and extends the author's earlier projects [easy-wakeword-trainer](https://github.com/mylegitches/easy-wakeword-trainer) and [piper-voice-helper](https://github.com/mylegitches/piper-voice-helper).

**Repository:** https://github.com/mylegitches/speech-model-toolkit
