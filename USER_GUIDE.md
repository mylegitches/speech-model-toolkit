# Speech Model Toolkit: User Guide

Step-by-step instructions for everything the app does. For installation details and the technical background see the [README](README.md); for a feature overview see [FULL_WRITEUP.md](FULL_WRITEUP.md).

**Contents**

1. [Getting started](#1-getting-started)
2. [Train a wake word](#2-train-a-wake-word)
3. [Create a voice](#3-create-a-voice)
4. [Build the dataset](#4-build-the-dataset)
5. [Train the voice](#5-train-the-voice)
6. [Listen and install in Home Assistant](#6-listen-and-install-in-home-assistant)
7. [Test Lab: wake word + voice + AI](#7-test-lab-wake-word--voice--ai)
8. [Settings](#8-settings)
9. [Using it on your phone](#9-using-it-on-your-phone)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Getting started

1. Start the app on the computer or server that will do the training:
   ```bash
   docker compose up -d --build
   ```
   No NVIDIA GPU? Use `docker compose -f docker-compose.cpu.yml up -d --build` instead.
2. Open **http://localhost:8765**, or the server's address.
3. The **Home** tab shows what's ready: wake word training data, voices, the GPU, and whether an AI is connected. A pulsing dot on a tab means something is training there.

The tabs keep working in the background. You can start a voice training, switch to another tab, even close the browser; come back later and the progress is still there.

> **Microphone:** browsers only allow it on `https://` addresses or `http://localhost`. To use the microphone from another computer or a phone, serve the app over HTTPS (for example with Nginx Proxy Manager, see [Troubleshooting](#10-troubleshooting)).

---

## 2. Train a wake word

**What you get:** a small model that listens for a phrase like "hey jarvis", as `.onnx` and `.tflite` files.

1. Open the **Wake Word** tab.
2. **First time only:** click **Download Data**. It downloads about 5–20 GB of training material. It can take a while; if the connection drops, click it again and it resumes.
3. **1. Wake phrase:** type the phrase (letters, numbers and spaces, up to 6 words) and click **🔊 Preview** to hear how it will be spoken.
   - Pick something 2–4 syllables long that isn't said in normal conversation ("hey jarvis" is better than "okay").
4. **2. Train:** click **Train** and watch the four stages: Generate → Augment → Train → Export. About 7–10 minutes; a GPU isn't needed.
5. When it's done, download the **.zip** (both formats) or just **.onnx** / **.tflite**.
6. **3. Test:** pick the model and your microphone, click **Start Listening** and say the phrase. The ring lights up green when it's detected. If it triggers too easily or too rarely, change **Wake word sensitivity** in Settings → Audio.

**Using it in Home Assistant:** copy the `.tflite` file into the `/share/openwakeword` folder of Home Assistant (the openWakeWord add-on picks up custom models there), restart the add-on, and choose the wake word in **Settings → Voice assistants**.

---

## 3. Create a voice

1. Open the **Voice** tab.
2. **1. Voice:** click **+ New voice**, give it a name (e.g. `dad`), choose the language and whether it should sound female or male, then click **Create voice**.

Each voice has its own dataset, training and exported versions. Switch between voices with the dropdown.

---

## 4. Build the dataset

**2. Dataset** is the training data: short clips of the voice, each with exactly what was said. More clips mean a better voice:

| Clips | Result |
|---|---|
| 50 | The minimum to train. Rough, but recognisable. |
| 300+ | Sounds much better. |
| 1,000 | Excellent. |

The readiness bar shows where you are. Use any of the three tabs and mix them freely.

### Read sentences

1. Choose your microphone and click **🎤 Enable microphone**. Speak normally and watch the level bar: green is good, red means too loud.
2. Read the sentence shown, naturally and clearly.
   - **R** starts/stops recording (or the ● Record button)
   - **P** plays it back
   - **S** saves it and shows the next sentence
   - **K** skips a sentence you don't want to read

**Tips:** use the same microphone and room every session, keep a steady distance from the mic, and avoid background noise (fridges, fans, TV).

### Speak freely

Talk instead of reading: tell a story, describe your day, read a book aloud.

1. Enable the microphone (as above) and choose **Background noise**: *Reduce* is right for most rooms.
2. Click **● Start recording**, talk for as long as you like (up to 30 minutes), then click **■ Stop and transcribe**.
3. Wait while it's transcribed. A progress percentage shows while it works.
4. **Review the clips** (see below).

Pause briefly between sentences: it helps the app cut clean clips.

### Import file

Use recordings you already have: voice memos, podcasts, interviews, movies or a whole series of episodes. Audio or video up to 8 GB per file (MP4, MKV, AVI, MOV, WMV, MP3, WAV, FLAC and more). For videos only the sound is used.

1. Choose what to import:
   - **Files**: select one or several.
   - **📁 Choose folder**: a whole folder, including everything in its subfolders.
   - Or **drag** files or folders onto the Import tab.

   A list shows the audio and video files found (subtitles, images and other files are skipped), with their sizes. Untick any you don't want.
2. Choose **Background noise**. For videos it switches to *Remove* automatically (it also removes music and effects).
3. If several people talk in it, tick **Several people are talking**. This is ticked automatically for videos.
4. Click **Import** (or **Import N files**). The files upload one after another with a percentage, then are processed one at a time; the others show *Waiting for the other files to finish…*.
5. **If the file has several audio tracks** (for example English, Spanish and a director's commentary), you're asked which one to use. The one in the voice's language is preselected.
   - For surround sound (5.1/7.1), **Dialogue only** is on by default: it uses just the center channel, where movie dialogue is, leaving most music and effects out.
6. **If several people talk**, the review starts with **Who do you want?**: one card per person, with how long they talk, three ▶ samples and a quote.
   - Play the samples and click **Use this voice** on the right person.
   - If the voice already has recordings, the person who sounds most like them is marked **Sounds like you** and already selected.
   - If one person was split into two cards (shouting vs whispering, phone vs studio), select both.
7. **Review the clips** (see below).

#### A series of videos with the same people

Every file is compared with the people found in your earlier files, so the same person keeps the same name everywhere: their card in each file says *also in N other files*. Above the takes, **People in your files** lists everyone found so far:

- **Click a name** to rename that person (e.g. "Jerry"), in the panel or on a speaker card in a file; Enter saves, Esc cancels. The name then shows in every file. Giving two people the same name offers to merge them.
- **This is the voice** marks the person you want. They're preselected in every file already imported and every file you import later.
- **Maybe the same as …** appears when two entries sound alike (one person split in two, e.g. shouting vs. calm). ▶ to compare, then **Merge** or **Not the same** (the suggestion goes away for good).
- **Tick two or more cards** and **Merge selected** to combine them (they go into the one with the most speech), or use **Same as…** on a card.
- With many people, **Find a person…** searches by name, and people with little speech are behind **Show all**.
- **▶** plays one of their clips.

**Let AI name them (optional).** With an AI connection and **Speaker identification** turned on in Settings, the AI reads a few lines each person said and works out who they are, for a whole series too:

- It looks up the cast on TVmaze. Put the show in a folder named after it (`The Sopranos/Season 1/S01E01.mkv`), or include the IMDb ID in a file name (`… tt0141842.mkv`). Folders like *Season 1* or *Disc 2* are ignored.
- With **web search** on and a provider that supports it (OpenRouter, Perplexity, Gemini, Anthropic, OpenAI search models), it can also look things up online.
- Each card then shows its answer, e.g. *AI: Carmela Soprano (Edie Falco) · high*; hover for the reason, and **Use this name** to accept it.
- With **Name cards automatically**, "Person N" cards get the name when the AI is sure. Names you typed are never changed.
- With **Merge cards automatically**, two cards the AI is sure are the same character are merged, but only if their voices are also somewhat alike. Otherwise they show up as *Maybe the same as …: AI thinks both are …* for you to decide.
- It runs after every imported file. **🔎 Identify with AI** asks about the people it hasn't identified yet; Shift-click asks about everyone again.

Once the person you want is marked, review the clips of each file, then use **✓ Save *name*'s clips from all files** to add all of them to the dataset at once. Unticked clips stay out.

> Only use recordings of people who agreed to have their voice cloned.

**Already have a prepared dataset?** Under **Import file**, open *Already have a prepared dataset?* and import a zip with `metadata.csv` (`file|text` per line) plus the audio, or audio files each next to a `.txt` with its transcript. These clips are added as they are.

**Export the dataset.** **⬇ Export dataset (.zip)**, next to the clip count, downloads all saved clips: `metadata.csv` (`id|text` per line) and the audio in `wavs/`, the LJSpeech layout that Piper and most TTS trainers use. Use it as a backup, to train elsewhere, or to import into another voice here.

### Reviewing clips

Spoken and imported takes wait for your review; nothing is added until you save. For every clip:

- **▶** listens to it.
- **Fix the text** so it matches *exactly* what was said, word for word. This matters a lot for quality.
- **Untick** clips with mistakes, coughs, laughter, music, other voices or people talking over each other.
- Change **Noise** for the whole take and listen again, if needed.

Then click **✓ Save N clips to the dataset** (at the top or bottom of the list) to add them, or **Discard** to throw the take away. **Clips only count once saved**: until then the Dataset counter shows them as *waiting for review*, and the Train step reminds you. Unfinished reviews are kept, so you can come back later.

---

## 5. Train the voice

**3. Train** fine-tunes a pretrained voice on your dataset.

1. **Starting voice:** leave **Auto-detect (recommended)** selected. When training starts it compares your clips with every pretrained voice for your language and starts from the one that sounds most like you.
   - Click **Find match** to see the ranking first (each voice gets a similarity %).
   - Click **▶** next to any voice to hear it, and select one yourself if you prefer.
2. **Pick how long to train:**

   | Preset | Time | |
   |---|---|---|
   | Quick test | 30 min | rough, to check that everything works |
   | **Good** | 3 hours | the default |
   | Best | 8 hours | |
   | Until I stop it | no limit | press **■ Stop** when it sounds right |

3. Click **Train**. The progress bar, stage and log show what's happening. You can leave the page.
4. **Train more** continues where the last run stopped: add more clips and train again to improve the voice.

**Advanced settings:** starting voices from other languages, your own checkpoint, training from scratch (very slow), hours, epochs, batch size (lower it if the GPU runs out of memory), device and sample rate.

A GPU makes a huge difference: hours instead of days. If voice training runs on the CPU, the Home tab says *CPU only*.

---

## 6. Listen and install in Home Assistant

**4. Test & install**

1. Pick a **Version** (each export is a snapshot of the training), type any text and click **🔊 Speak**.
2. While training runs, **Export latest version now** lets you hear how it's going.
3. Click **Download for Home Assistant (.zip)**.
4. In Home Assistant, open **Settings → Add-ons → Piper → Open Web UI** and upload the `.onnx` and `.onnx.json` files from the zip, or copy them to `/share/piper`.
5. Restart the Piper add-on if the voice doesn't appear.
6. Go to **Settings → Voice assistants**, edit your assistant and choose the new voice.

---

## 7. Test Lab: wake word + voice + AI

Try your wake word and voice together, as a real assistant.

1. Open **Test Lab**.
2. **1. Setup:** choose a **wake word** and a **reply voice**. The line below says whether replies come from an AI or are the fixed confirmation.
3. **2. Listen:** click **🎤 Start listening** and say the wake word.
   - **No AI connection:** you hear *"Your voice wakeword model was trained and triggered successfully."* in the chosen voice.
   - **With an AI connection:** after the ring lights up, ask your question and pause. It transcribes you, asks the AI and speaks the answer. Ask follow-up questions the same way; it remembers the conversation.
4. **3. Conversation** shows everything that was said. ▶ replays a reply, ⬇ downloads it (if allowed in Settings). You can also type a question in the box. **New conversation** clears the history.

The status pill shows what it's doing: *Listening → Heard it! → Your turn → Transcribing → Thinking → Speaking*.

---

## 8. Settings

### AI connections

1. Click **+ Add connection** and choose a provider:
   - **Cloud:** OpenAI, Anthropic, Google Gemini, OpenRouter, Ollama Cloud, Groq, Mistral AI, DeepSeek, xAI, Together AI, Fireworks AI, Cerebras, Cohere, Perplexity.
   - **On your own machines:** Ollama (local), LM Studio, or any OpenAI-compatible server (Custom).
2. The **Base URL** fills in by itself. Paste your **API key** ("Get a key ↗" opens the provider's key page). Local servers don't need a key.
3. Click **Load models** and pick one from the list (or type the model name).
4. Click **Test connection**. A green **✓ Connected** shows the model's reply and how fast it answered.
5. Click **Save**. The first saved connection becomes active; with several, click **Use** on the one the Test Lab should use.

For a local Ollama on the same machine as the toolkit, the default URL works. For Ollama on another computer, use that computer's address (e.g. `http://192.168.1.50:11434`) and start Ollama with `OLLAMA_HOST=0.0.0.0`.

Keys are stored only on your server and are never shown again (you'll see `sk-…abcd`).

### Assistant

- Which connection answers (or the fixed reply)
- The **system prompt**: by default it asks for short answers that sound good spoken aloud
- Temperature and maximum reply length
- The fixed reply text

### Speaker identification (AI)

For imported files (see *A series of videos with the same people*). Needs an active AI connection.

- **Identify speakers with AI**: on/off.
- **Look up the cast on TVmaze**: from an IMDb ID in the file names or the show's folder name.
- **Let the AI search the web**, where the provider supports it.
- **Name "Person N" cards automatically** when the AI is sure.
- **Merge cards automatically** when the AI is sure they're the same character and their voices are alike.

File names and a few transcribed lines per person are sent to your AI provider, and the show name or IMDb ID to TVmaze.

### Speech recognition

The Whisper model used to understand speech: **Base (English)** is a good default. *Small* is more accurate but slower; *multilingual* models are needed for other languages. **Download model now** avoids a wait the first time.

### Audio

- **Microphone** and **Speaker:** used in every tab and remembered by your browser. Use 🎤 Test / 🔊 Test to check them.
- **Volume** of replies and previews.
- **Wake word sensitivity:** lower triggers more easily.
- **Pause after each reply:** stops the assistant from re-triggering right away.
- **Echo cancellation / noise suppression / automatic gain** for listening. Voice recordings for the dataset always use the raw microphone.
- **Allow downloading generated .wav files.**

---

## 9. Using it on your phone

1. Open the app's **https://** address on your phone.
2. On the Home tab, follow **Use it like an app**:
   - **iPhone:** tap Share → *Add to Home Screen*.
   - **Android:** tap *Install*, or choose *Add to Home screen* in the browser menu.
3. It now opens full screen, with the tabs at the bottom.

The screen stays on while the Test Lab or the wake word tester is listening. Recording dataset sentences works too; use a headset microphone for the best quality.

---

## 10. Troubleshooting

Errors are shown where they happen, with what to do. For more detail, look at the server log:

```bash
docker compose logs -f toolkit
```

| Problem | What to do |
|---|---|
| **"The microphone only works on HTTPS…"** | Open the app via `https://` or `http://localhost`. |
| **"Microphone access is blocked"** | Click the icon next to the address bar and allow the microphone. |
| **"The file is too large… reverse proxy"** | Raise the proxy's upload limit (see below). |
| **"The connection to the server dropped"** in the Test Lab | Enable **Websockets Support** on your reverse proxy. |
| **Live logs stop updating** | Turn off proxy buffering (see below); refreshing the page also works. |
| **Train says the dataset has 0 (or very few) clips** after importing or recording a take | The take's clips aren't saved yet. In **2. Dataset**, open the take (the counter shows *N clips waiting for review*), pick the speaker if asked, and press **✓ Save N clips to the dataset**. |
| **"The GPU ran out of memory"** | Lower **Batch size** in Advanced settings, or close other programs using the GPU. |
| **"The server's disk is full"** | Free space in the `data` folder's disk (old takes, voices you don't need). |
| **Wake word triggers too often / too rarely** | Adjust **Wake word sensitivity** in Settings → Audio, or retrain with a longer phrase. |
| **The voice sounds robotic** | Add more clips (300+), check transcripts are exact, train longer. |
| **Wrong language picked from a movie** | Discard the take and import again; choose the right track when asked. |
| **Two cards for the same person** | In **People in your files**: use **Merge** on the *Maybe the same as* suggestion, tick both and **Merge selected**, or give them the same name. In one file you can also just select both with **Use this voice**. |
| **A file failed or was interrupted by a restart** | Press **↻ Try again** on it: it's processed again from the file already uploaded. |
| **Folder import doesn't offer a folder button / drag and drop** | Use a desktop browser (Chrome, Edge, Firefox, Safari); phones can only pick single files. |
| **"This file has no audio track"** | The file only has video; use another copy. |

**Nginx Proxy Manager:** edit the proxy host → **Details**: switch on *Websockets Support* → **Advanced**, add:

```nginx
client_max_body_size 8g;
proxy_request_buffering off;
proxy_buffering off;
proxy_read_timeout 3600s;
proxy_send_timeout 3600s;
```

**Where your files are:** everything lives in the `data` folder next to `docker-compose.yml`:

- wake words: `data/wakeword-models/`
- voices: `data/voice/voices/`
- settings: `data/settings.json`

Back up that folder to keep your work.
