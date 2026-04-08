# Fake It Till You Make It — Implementation Plan

> A terminal-based voice AI conversation system with semantic pause detection and backchannel fillers, inspired by [Unmute](https://github.com/kyutai-labs/unmute).

---

## Goal

Build a **terminal-only** voice conversation system where you speak into your microphone and an AI responds in real-time — with natural conversational behavior:
- Knows **when you've finished** speaking (semantic pause detection)
- Knows **when you're mid-sentence** (doesn't interrupt)
- Makes small acknowledgment sounds ("mhm", "yeah") during your pauses to show it's listening
- Full interruption support — speak over the bot and it shuts up

---

## Tech Stack

| Component | Technology | Runs On | Memory |
|-----------|-----------|---------|--------|
| **STT** | `kyutai/stt-1b-en_fr` (DSM-ASR, 1B params) | GPU | ~2 GB VRAM |
| **TTS** | Kyutai Pocket TTS (DSM framework, runs on CPU!) | CPU | ~300 MB RAM |
| **LLM** | TBD — small Qwen model (~3-4B) via OpenAI-compatible API | GPU | ~3-4 GB VRAM |
| **Audio I/O** | `sounddevice` (PortAudio) | CPU | negligible |
| **Runtime** | Python 3.11.9, Debian 13 | — | — |
| **Total GPU** | STT + LLM | GPU | **~5-6 GB** ✅ (fits in 8GB) |

### Source Repositories

| Model | Repository | Package |
|-------|-----------|---------|
| STT + TTS | [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) | `pip install moshi>=0.2.6` |
| Unmute (reference only) | [kyutai-labs/unmute](https://github.com/kyutai-labs/unmute) | in `./unmute/` dir (gitignored) |

### Why `sounddevice` for Audio

| | `sounddevice` | `pyaudio` |
|---|---|---|
| **API** | Modern, numpy-native, callback + stream | C-style, manual buffer management |
| **Install** | `pip install sounddevice` (bundles PortAudio) | Needs `portaudio19-dev` system package |
| **Threading** | Built-in async-friendly callbacks | Manual threading required |
| **Maintenance** | Actively maintained | Stale |

---

## Architecture

Unlike Unmute which uses separate Rust servers for STT/TTS communicating via WebSocket + msgpack, we run **everything in a single process** — STT and TTS models loaded directly in Python, LLM accessed via HTTP API.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SINGLE PYTHON PROCESS                        │
│                                                                     │
│  ┌────────────┐     ┌────────────────────────────────────────────┐  │
│  │ MICROPHONE │──►  │              ORCHESTRATOR                  │  │
│  │ sounddevice│     │                                            │  │
│  │ InputStream│     │  Audio frames → STT model (GPU)            │  │
│  │ (C thread) │     │       │                                    │  │
│  └────────────┘     │       ├─► pause_prediction (EMA smoothing) │  │
│                     │       ├─► words → chat_history              │  │
│                     │       │                                    │  │
│  ┌────────────┐     │  pause detected?                           │  │
│  │  SPEAKER   │◄──  │       ├─► YES + done → flush → LLM → TTS  │  │
│  │ sounddevice│     │       ├─► YES + mid-sentence → filler!     │  │
│  │ OutputStream│    │       └─► NO → continue listening          │  │
│  │ (C thread) │     │                                            │  │
│  └────────────┘     └────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│                     ┌──────────────┐       ┌──────────────┐        │
│                     │  LLM Server  │       │  TTS Model   │        │
│                     │  (external)  │       │  (in-process)│        │
│                     │  HTTP API    │       │  Pocket TTS  │        │
│                     │  (GPU)       │       │  (CPU)       │        │
│                     └──────────────┘       └──────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Unmute

| Aspect | Unmute | Ours |
|--------|--------|------|
| STT | Rust server via WebSocket + msgpack | **Direct PyTorch in-process** — no network overhead |
| TTS | Rust server via WebSocket + msgpack | **Direct PyTorch in-process** — no network overhead |
| LLM | vLLM server (OpenAI API) | Same — external OpenAI-compatible API |
| Audio | Browser mic via WebRTC + Opus codec | **Direct sounddevice** — raw PCM, zero encoding |
| UI | Next.js web frontend | **Terminal** (rich text with `rich`) |
| Concurrency model | All async I/O (WebSocket sends) | **Dedicated worker threads** for model inference |
| Echo cancellation | Browser WebRTC AEC (free) | **Headphones** (see below) |

---

## Full-Duplex Design & Known Constraints

### ✅ How Full-Duplex Works In Our Design

The mic is **always recording**, even while the bot speaks. This works because `sounddevice` uses C-level PortAudio threads that are completely independent from Python:

```
Thread 1 (PortAudio C): Mic callback → pushes frames to mic_queue      [ALWAYS RUNS]
Thread 2 (PortAudio C): Speaker callback → pulls from speaker_queue     [ALWAYS RUNS]
Thread 3 (dedicated):   STT worker — holds streaming context, GPU       [LONG-RUNNING]
Thread 4 (dedicated):   TTS worker — holds streaming context, CPU       [LONG-RUNNING]
Main thread:            asyncio event loop — orchestrates via queues     [NEVER BLOCKS]
```

STT (GPU) and TTS (CPU) don't compete for resources. Both release the GIL during compute-heavy operations, allowing true parallel execution.

### 🎧 Echo Cancellation: Headphones Required

**The only real constraint:** Without browser-level AEC, the speaker audio would get picked up by the mic, causing the STT to transcribe the bot's own voice → infinite interruption loop.

**Solution: Wear headphones.** This completely eliminates the echo path.

```
WITH headphones:                      WITHOUT headphones (BROKEN):
Mic ──► STT (hears only user)         Mic ──► STT (hears user + bot echo!)
Speaker ──► Headphones (user only)     Speaker ──► Room ──► Mic (feedback loop)
```

Speaker support can be added later via:
- **PipeWire AEC** (`pactl load-module module-echo-cancel`) — OS-level, decent quality
- **Software AEC** (`speexdsp`) — in-code, more complex

### ⚡ Critical Architecture Rule: Dedicated Worker Threads

The STT and TTS models use `streaming()` context managers that hold internal state (KV-cache, codec buffers). These contexts **must stay alive for the entire session**. This means we can't use per-frame `asyncio.to_thread()` — we need **dedicated long-running worker threads**:

```python
# ❌ BAD — streaming context dies after each call
async def process_frame(frame):
    return await asyncio.to_thread(stt_model.process, frame)  # Context lost!

# ✅ GOOD — dedicated thread holds context alive for entire session
def stt_worker(mic_queue: Queue, result_queue: Queue):
    with mimi.streaming(1), lm_gen.streaming(1):    # Context lives for entire conversation
        while running:
            frame = mic_queue.get()                  # Block until frame arrives
            tokens = mimi.encode(frame)              # Audio → Mimi tokens
            text, vad = lm_gen.step_with_extra_heads(tokens)  # → text + VAD
            result_queue.put((text, vad))             # Send to main loop
```

This pattern is actually **simpler** than per-call threading — it's a long-running worker with queue-based communication. Same concept as Unmute's `_stt_loop()`, just without the WebSocket.

---

## Verified `moshi` Package API

We have verified the exact Python API by reading Kyutai's official scripts. The `moshi` package **does** provide direct frame-by-frame streaming inference — no servers needed.

### STT API (from `stt_from_file_pytorch.py`)

```python
import moshi.models
import torch

# 1. Load models (one-time setup)
info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
mimi = info.get_mimi(device="cuda")                          # Mimi audio codec
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)     # STT transformer
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

# 2. Process audio frame-by-frame (streaming)
with mimi.streaming(1), lm_gen.streaming(1):        # ← Context must stay alive!
    for audio_chunk in audio_frames:                  # Each = [1, 1, 1920] tensor
        audio_tokens = mimi.encode(audio_chunk)
        text_tokens, vad_heads = lm_gen.step_with_extra_heads(audio_tokens)
        
        pr_vad = vad_heads[2][0, 0, 0].cpu().item()   # ← Our pause_prediction!
        text_token = text_tokens[0, 0, 0].cpu().item()
```

### TTS API (from `tts_pytorch_streaming.py`)

```python
from moshi.models.tts import TTSModel, script_to_entries
import sounddevice as sd

# 1. Load model
checkpoint_info = CheckpointInfo.from_hf_repo(hf_repo)
tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device="cpu")
condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)

# 2. Create generator + stream text → audio
def on_frame(frame):
    pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
    speaker_queue.put(np.clip(pcm[0, 0], -1, 1))

gen = TTSGen(tts_model, [condition_attributes], on_frame=on_frame)
with tts_model.mimi.streaming(1):                    # ← Context must stay alive!
    entries = prepare_script(tts_model, text, first_turn=True)
    for entry in entries:
        gen.append_entry(entry)
        gen.process()                                  # Generates audio via on_frame callback
    gen.process_last()                                 # Flush remaining audio
```

**Key finding:** Both APIs use `streaming()` contexts that hold model state. These contexts must persist for the entire conversation — this is why we need **dedicated worker threads**, not per-call `to_thread()`.

---

## Module Plan

### File Structure

```
fake-it-till-you-make-it/
├── PLAN.md                  # This file
├── README.md
├── requirements.txt
├── main.py                  # Entry point — start conversation
│
├── core/
│   ├── __init__.py
│   ├── orchestrator.py      # The brain — audio loop, state machine, pause detection
│   ├── audio_io.py          # Microphone input + speaker output via sounddevice
│   ├── conversation.py      # Chat history + conversation state machine
│   └── config.py            # All constants and configuration
│
├── stt/
│   ├── __init__.py
│   ├── stt_engine.py        # Kyutai STT model wrapper — dedicated worker thread
│   └── ema.py               # ExponentialMovingAverage (from Unmute)
│
├── llm/
│   ├── __init__.py
│   ├── llm_client.py        # OpenAI-compatible streaming client
│   └── system_prompt.py     # System prompt template
│
├── tts/
│   ├── __init__.py
│   └── tts_engine.py        # Pocket TTS model wrapper — dedicated worker thread
│
├── fillers/
│   ├── __init__.py
│   ├── filler_manager.py    # Backchannel detection + clip selection + cooldown
│   └── clips/               # Pre-generated filler audio clips (WAV)
│       ├── mhm_1.wav
│       ├── mhm_2.wav
│       ├── yeah_1.wav
│       ├── right_1.wav
│       └── ...
│
└── utils/
    ├── __init__.py
    └── timer.py             # Stopwatch utility
```

---

## Phase-by-Phase Implementation

### Phase 0: Environment Setup
- [ ] Create venv, install `sounddevice`, `numpy`, `moshi>=0.2.6`, `torch`, `openai`, `rich`
- [ ] Install system dependency: `sudo apt install libportaudio2`
- [ ] Verify `sounddevice` can capture/playback on the machine
- [ ] Verify `moshi` package can load `kyutai/stt-1b-en_fr` model
- [ ] Verify `moshi` package can load Pocket TTS model
- [ ] Set up LLM server (Ollama or similar) — not blocking, can use dummy LLM initially

### Phase 1: Audio I/O Layer (`core/audio_io.py`)
Build the lowest level first — can we hear ourselves?

- [ ] `MicrophoneInput` class — opens an `InputStream` with sounddevice
  - Sample rate: 24,000 Hz (Mimi codec requirement)
  - Frame size: 1,920 samples (80ms, matching Kyutai's FRAME_TIME_SEC)
  - Callback puts frames into an `asyncio.Queue`
  - Handles device selection (list available devices)
- [ ] `SpeakerOutput` class — opens an `OutputStream` with sounddevice
  - Reads from an `asyncio.Queue`
  - Plays PCM float32 audio
  - Supports interruption (clear buffer instantly)
- [ ] **Test**: Record from mic → immediately play back (echo test)

### Phase 2: STT Engine (`stt/stt_engine.py`)
Wrap Kyutai's STT as a dedicated worker thread with long-lived streaming context.

- [ ] Load `kyutai/stt-1b-en_fr` model via `moshi` package:
  ```python
  info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
  mimi = info.get_mimi(device="cuda")
  lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
  lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
  ```
- [ ] `STTEngine` class — **dedicated worker thread pattern**:
  ```python
  class STTEngine:
      def __init__(self, device="cuda"):
          # Load models...
          self.mic_queue = queue.Queue()      # Main loop puts frames here
          self.result_queue = queue.Queue()   # Worker puts results here
          self.thread = threading.Thread(target=self._worker, daemon=True)
      
      def _worker(self):
          """Long-running thread — streaming context lives for entire session."""
          with self.mimi.streaming(1), self.lm_gen.streaming(1):
              while self.running:
                  frame = self.mic_queue.get()                   # Blocks until frame
                  tokens = self.mimi.encode(frame)               # Audio → tokens
                  text, vad = self.lm_gen.step_with_extra_heads(tokens)
                  pr_vad = vad[2][0, 0, 0].cpu().item()          # Pause probability
                  text_token = text[0, 0, 0].cpu().item()
                  self.result_queue.put(STTResult(text_token, pr_vad))
      
      def feed_frame(self, frame: np.ndarray):
          """Called from main loop — non-blocking."""
          self.mic_queue.put(frame)
      
      async def get_result(self) -> STTResult:
          """Async wrapper to read from result queue without blocking event loop."""
          return await asyncio.get_event_loop().run_in_executor(None, self.result_queue.get)
  ```
- [ ] Handle the 12-frame warm-up period (skip first ~960ms, same as Unmute)
- [ ] Track `current_time` (frame counter × 0.08s)
- [ ] Decode text tokens → words using the tokenizer from `info.get_text_tokenizer()`
- [ ] `ema.py` — copy `ExponentialMovingAverage` from Unmute (it's only 38 lines)
  - Attack half-life: 0.01s, Release half-life: 0.01s
  - Update with `prs[2]` (= `vad_heads[2]`) from each STT step
- [ ] **Benchmark**: Measure inference time per frame — must be <80ms
- [ ] **Test**: Speak into mic → see words printed in terminal in real-time

### Phase 3: Conversation State Machine (`core/conversation.py`)
Direct port from Unmute's `chatbot.py`.

- [ ] `Conversation` class:
  - `chat_history: list[dict]` — `[{role, content}, ...]`
  - `conversation_state() -> "waiting_for_user" | "user_speaking" | "bot_speaking"`
  - `add_message_delta(text, role) -> is_new_message`
  - `preprocessed_messages()` — clean up for LLM (merge, strip interruptions, etc.)
- [ ] State transitions (same as Unmute):
  ```
  waiting_for_user → (first word) → user_speaking
  user_speaking → (pause detected) → bot_speaking  
  bot_speaking → (user interrupts) → user_speaking
  bot_speaking → (response done) → waiting_for_user
  ```

### Phase 4: Basic Orchestrator (`core/orchestrator.py`)
The main loop — tie audio + STT + state machine together (no LLM/TTS yet).

- [ ] Main async loop:
  ```python
  async def run(self):
      stt_engine.start()  # Launches worker thread
      
      while running:
          # 1. Get audio frame from mic (arrive every 80ms)
          frame = await mic_queue.get()
          
          # 2. Feed to STT worker (non-blocking put)
          stt_engine.feed_frame(frame)
          
          # 3. Get result from STT worker (async, doesn't block event loop)
          result = await stt_engine.get_result()
          
          # 4. Update EMA with pause probability
          ema.update(dt=0.08, new_value=result.pr_vad)
          
          # 5. Process transcribed words
          if result.word:
              conversation.add_message_delta(result.word, "user")
              print(f"[USER] {result.word}", end=" ", flush=True)
          
          # 6. Check pause detection
          if determine_pause(ema.value, conversation.state):
              print("\n[SYSTEM] Pause detected!")
  ```
- [ ] `determine_pause()` — same logic as Unmute:
  - State must be `"user_speaking"`
  - EMA value > 0.6
- [ ] Flush logic — push silence frames through STT to get buffered words
  - `num_frames = ceil(0.5 / 0.08) + 1 = 8` silence frames
  - Feed silence tensor: `torch.zeros((1, 1, 1920), device="cuda")`
- [ ] Long silence detection — if `waiting_for_user` for >7 seconds, trigger `"..."` message
- [ ] **Test**: Speak naturally → see pause detection fire correctly in terminal

### Phase 5: LLM Integration (`llm/llm_client.py`)
Connect to an OpenAI-compatible server.

- [ ] `LLMClient` class:
  - `async stream_response(messages) -> AsyncIterator[str]`
  - Uses `openai.AsyncOpenAI` with configurable `base_url`
  - Supports streaming (token-by-token)
  - `rechunk_to_words()` — same as Unmute, re-chunk tokens into whole words
- [ ] System prompt (`system_prompt.py`):
  - Keep it simple — conversational, spoken format, no markdown
  - Include silence handling (`"..."` = user is silent)
  - Include interruption marker (`"—"` = bot was interrupted)
- [ ] Wire into orchestrator:
  - On pause → flush STT → call LLM → stream words
  - Print `[BOT]` text to terminal as it arrives
- [ ] **Test**: Full STT → LLM loop (text only, no TTS yet)

### Phase 6: TTS Engine (`tts/tts_engine.py`)
Wrap Kyutai's Pocket TTS as a dedicated worker thread.

- [ ] Load Pocket TTS model via `moshi` package:
  ```python
  from moshi.models.tts import TTSModel
  checkpoint_info = CheckpointInfo.from_hf_repo(tts_hf_repo)
  tts_model = TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device="cpu")
  condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
  ```
  - Pocket TTS runs on **CPU** (300MB) — does not compete with STT/LLM for GPU
- [ ] `TTSEngine` class — **dedicated worker thread pattern**:
  ```python
  class TTSEngine:
      def _worker(self):
          """Long-running thread — mimi streaming context lives for entire session."""
          gen = TTSGen(self.tts_model, [self.condition_attributes], on_frame=self._on_frame)
          with self.tts_model.mimi.streaming(1):
              while self.running:
                  text = self.text_queue.get()          # Block until text arrives
                  entries = prepare_script(self.tts_model, text, first_turn=...)
                  for entry in entries:
                      gen.append_entry(entry)
                      gen.process()                      # Generates audio via _on_frame
                  gen.process_last()                     # Flush remaining
      
      def _on_frame(self, frame):
          """Callback — sends audio to speaker queue."""
          if (frame != -1).all():
              pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
              self.speaker_queue.put(np.clip(pcm[0, 0], -1, 1))
  ```
  - Support cancellation (for interruption): set a flag that `_on_frame` checks
- [ ] **Benchmark**: Measure real-time factor — must synthesize faster than playback speed
- [ ] Wire into orchestrator:
  - LLM words → TTS text_queue → worker generates audio → speaker_queue
  - Audio plays as it's generated (no waiting for full response)
- [ ] **Test**: Type text → hear it spoken

### Phase 7: Interruption Handling
Make it possible to talk over the bot.

- [ ] **STT-word interruption**: If a word is transcribed while state is `"bot_speaking"`:
  - Cancel LLM generation task
  - Cancel TTS generation task
  - Clear speaker output buffer
  - Append `"—"` to assistant message
  - State → `"user_speaking"`
- [ ] **VAD interruption**: If `pause_prediction < 0.4` while `"bot_speaking"` (after 3s grace period):
  - Same cancel flow
- [ ] **Interruption marker in chat history**: Use `"—"` like Unmute so the LLM knows it was cut off
- [ ] Implementation detail: Use `asyncio.Task.cancel()` for LLM/TTS tasks, replace output queue (same trick as Unmute — old task writes to dead queue)
- [ ] **Test**: Bot is speaking → interrupt by talking → bot stops immediately

### Phase 8: Filler/Backchannel System (NEW — not in Unmute!)
The novel part — make the bot sound like it's actively listening.

- [ ] **Pre-generate filler clips** (`fillers/clips/`):
  - Use Pocket TTS to synthesize: "mhm", "yeah", "okay", "right", "uh-huh"
  - Generate 2-3 variants of each (different intonation via temperature)
  - Store as WAV files (short, ~300-500ms each)
  - Lower volume (70% of normal) so they feel like background acknowledgment
- [ ] **`FillerManager` class** (`fillers/filler_manager.py`):
  ```python
  class FillerManager:
      def __init__(self):
          self.clips = load_all_clips()      # Pre-loaded numpy arrays
          self.cooldown_sec = 6.0            # Minimum gap between fillers
          self.last_filler_time = 0
          self.min_speaking_time = 3.0       # Don't filler in first 3 seconds
      
      def should_play_filler(self, ema_value, state, time_speaking) -> bool:
          """Is this a mid-sentence pause where a filler would be natural?"""
          return (
              state == "user_speaking"
              and 0.3 <= ema_value <= 0.55        # Mid-pause, NOT end-of-sentence
              and time_speaking > self.min_speaking_time
              and (current_time - self.last_filler_time) > self.cooldown_sec
          )
      
      def get_random_clip(self) -> np.ndarray:
          """Return a random filler clip, avoiding repeats."""
          ...
  ```
- [ ] **Wire into orchestrator** — inside the main audio loop, after EMA update:
  ```python
  if filler_manager.should_play_filler(ema.value, state, time_speaking):
      clip = filler_manager.get_random_clip()
      await speaker.play_clip(clip)  # Non-blocking, short clip
      filler_manager.last_filler_time = current_time
  ```
- [ ] **Note on echo:** Filler clips are ~300ms (4 frames). With headphones, no echo issue. Start with headphones, optimize later if needed.
- [ ] **⚠️ Filler EMA tuning note:** The 0.3–0.55 range is our starting guess based on Unmute's pause detection. The actual sweet spot depends on how the EMA behaves with real speech. During Phase 8 testing, log the EMA values while speaking naturally and adjust — the boundaries may need shifting by ±0.1.
- [ ] **Test**: Speak a long sentence with natural pauses → hear "mhm" / "yeah" interspersed

### Phase 9: Terminal UI (`main.py`)
Pretty terminal output using `rich`.

- [ ] Live-updating display with `rich.Live`:
  ```
  ╔══════════════════════════════════════════════════╗
  ║  🎤 Fake It Till You Make It                     ║
  ║  Status: Listening...                            ║
  ╠══════════════════════════════════════════════════╣
  ║                                                  ║
  ║  [YOU] Hey, I was thinking about going to the    ║
  ║       park today because the weather is nice     ║
  ║                                     [mhm] 🎵    ║
  ║       and maybe we could bring some food.        ║
  ║                                                  ║
  ║  [BOT] That sounds like a great idea! The        ║
  ║        weather has been really lovely lately—     ║
  ║                                                  ║
  ║  [YOU] Oh wait, actually I just remembered I     ║
  ║       have a meeting at three.                   ║
  ║                                                  ║
  ╠══════════════════════════════════════════════════╣
  ║  pause_prediction: ████████░░ 0.82               ║
  ║  state: user_speaking | time: 12.4s              ║
  ╚══════════════════════════════════════════════════╝
  ```
- [ ] Show real-time `pause_prediction` bar (EMA value 0.0–1.0)
- [ ] Show conversation state
- [ ] Show when fillers play (`[mhm] 🎵`)
- [ ] Show interruptions (`—`)
- [ ] Ctrl+C to quit gracefully

### Phase 10: Polish & Edge Cases
- [ ] Graceful shutdown — close audio streams, unload models
- [ ] Handle audio device errors (mic not found, etc.)
- [ ] Handle LLM server being down
- [ ] First response — bot greeting on startup
- [ ] Conversation ending — detect "bye" and exit
- [ ] Configurable thresholds via `config.py` or CLI args
- [ ] **Optional: PipeWire AEC for speaker mode** (no headphones):
  ```bash
  pactl load-module module-echo-cancel use_master=yes
  ```

---

## Key Constants

```python
# Audio
SAMPLE_RATE = 24_000              # Hz — required by Mimi codec
SAMPLES_PER_FRAME = 1_920         # 80ms frames
FRAME_TIME_SEC = 0.08             # 1920 / 24000

# STT
STT_DELAY_SEC = 0.5               # Built-in model delay
STT_WARMUP_FRAMES = 12            # Skip first ~960ms
FLUSH_FRAMES = 8                  # ceil(0.5 / 0.08) + 1

# Pause Detection (see UNDERSTAND_unmute.md for theory)
PAUSE_THRESHOLD = 0.6             # EMA value to trigger response
EMA_HALF_LIFE = 0.01              # Seconds (attack and release)

# Interruption
INTERRUPTION_VAD_THRESHOLD = 0.4  # Below this → user started speaking
UNINTERRUPTIBLE_TIME = 3.0        # Seconds grace period
INTERRUPTION_CHAR = "—"           # Em-dash in chat history

# Fillers (NEW)
FILLER_EMA_LOW = 0.3              # Below = user actively speaking
FILLER_EMA_HIGH = 0.55            # Above = probably end of sentence
FILLER_COOLDOWN = 6.0             # Seconds between fillers
FILLER_MIN_SPEAKING_TIME = 3.0    # Don't filler too early
FILLER_VOLUME = 0.7               # 70% of normal volume

# Silence
USER_SILENCE_TIMEOUT = 7.0        # Seconds before bot responds to silence

# LLM
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
```

---

## Dependencies

```
# requirements.txt
moshi>=0.2.6           # Kyutai's STT + TTS models (PyTorch)
torch                   # PyTorch backend
sounddevice             # Audio I/O
numpy                   # Audio processing
sphn                    # Audio file reading (used by moshi scripts)
julius                  # Audio resampling (used by moshi scripts)
openai                  # LLM client (OpenAI-compatible API)
rich                    # Terminal UI
```

**System packages** (Debian 13):
```bash
sudo apt install libportaudio2    # Required by sounddevice
```

> **Note:** `sphn` and `julius` are used in Kyutai's scripts. We may or may not need them
> depending on whether sounddevice already gives us audio at 24kHz. Verify in Phase 1.

---

## Implementation Order

```
Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 3 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
  setup     audio      stt     orchestr   state     llm        tts      interrupt  fillers    ui       polish
  ─────     ─────      ───     ────────   ─────     ───        ───      ─────────  ───────    ──       ──────
  30min     1hr        2hr     2hr        30min     1hr        2hr      2hr        3hr        1hr      2hr
```

**Total estimated: ~17 hours of focused work**

Critical path: **Audio → STT → Orchestrator** — once words come out of the STT with pause detection working, everything else builds on top.

---

## Open Questions

> [!IMPORTANT]  
> **LLM Choice**: Which Qwen model and how will you run it?
> - **Ollama** (easiest): `ollama run qwen2.5:3b` → API at `http://localhost:11434/v1/`
> - **llama.cpp** (fastest): Compile + run GGUF model
> - **vLLM** (most compatible): Same as Unmute uses
> 
> This doesn't block Phases 0-4. We can use a dummy LLM (returns fixed text) until you decide.

> [!NOTE]
> **Hardware**: Confirm your GPU model and VRAM. We need:
> - STT: ~2 GB VRAM (GPU)
> - TTS: ~300 MB RAM (CPU)
> - LLM: ~3-4 GB VRAM (GPU, if local)
> - Total GPU: ~5-6 GB → 8GB GPU is fine ✅
