# My Code — Full Understanding Guide

> This document walks you through every piece of your voice AI system.
> At each step, you'll find links to the exact lines of code so you can
> jump right in. Read this alongside the code — open the linked files
> and follow along.

---

## Table of Contents

1. [How to Read This Guide](#how-to-read-this-guide)
2. [The 10-Second Overview](#the-10-second-overview)
3. [Starting Up — What Happens When You Run `python main.py`](#starting-up)
4. [The Main Loop — 80ms at a Time](#the-main-loop)
5. [Audio Capture — From Mic Hardware to Python](#audio-capture)
6. [The STT Engine — The Neural Network Inside](#the-stt-engine)
7. [The Model — What's Actually Inside the 1B Parameter Brain](#the-model)
8. [Word Emission — Turning Tokens Into Words](#word-emission)
9. [Pause Detection — The Semantic VAD](#pause-detection)
10. [The Flush — Draining the Pipeline](#the-flush)
11. [The State Machine — Who's Talking?](#the-state-machine)
12. [Silence Detection — When Nobody Speaks](#silence-detection)
13. [Thread Architecture — Three Threads Working Together](#thread-architecture)
14. [Config — All the Knobs](#config)
15. [What's Next — Phases 5-8](#whats-next)

---

## How to Read This Guide

Every code reference is a clickable link. For example:

> Open [orchestrator.py → run()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L127-L190) to see the main loop.

Click that, and your editor will jump to exactly that code. Follow along!

---

## The 10-Second Overview

You speak into your mic. Your voice becomes text. The system detects when you've finished talking. It responds.

```
You speak → Mic records → GPU transcribes → Pause detected → Bot responds
   80ms        80ms          ~15ms             ~80ms           instant
```

All of this happens in real-time, 12.5 times per second.

---

## Starting Up

Everything begins in [main.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/main.py#L1-L39). It's tiny — just creates an `Orchestrator` and calls `run()`:

```python
async def main():
    orch = Orchestrator(device="cuda")
    await orch.run()
```

`asyncio.run(main())` creates the event loop that powers the whole system. Without it, `await mic.get_frame()` wouldn't work.

The Orchestrator's [start()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L88-L110) method fires up everything in order:

```
1. stt.load_model()     ← Downloads 1B param model, loads onto GPU (~30s first time)
2. stt.warmup()         ← Runs 15 dummy frames to compile CUDA kernels (~5s)
3. mic.start()          ← Opens microphone, C-thread callback starts firing every 80ms
4. speaker.start()      ← Opens speaker (plays silence until we feed it audio)
5. stt.start()          ← Launches GPU worker thread
```

After this, you see `🎤 Listening... (speak into your mic)`.

> 📂 **Go to:** [orchestrator.py → start()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L88-L110)

---

## The Main Loop

The heart of everything is [orchestrator.py → run()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L127-L190). It's an infinite loop where each iteration takes ~80ms:

```
┌──────────────────────────────────────────────┐
│             One Loop Iteration (~80ms)        │
│                                               │
│  1. await mic.get_frame()    ← waits ~80ms   │
│  2. stt.feed_frame(frame)   ← instant        │
│  3. drain_results()         ← process words  │
│  4. if normal_mode:                           │
│       check_silence()                         │
│       determine_pause()                       │
│     elif flush_mode:                          │
│       check if flush done                     │
└──────────────────────────────────────────────┘
                    │
                    ▼
              (repeat forever)
```

**Step 1** — The loop is paced by the microphone. `await mic.get_frame()` blocks until the next 80ms chunk of audio arrives. This is what makes the loop run exactly 12.5 times per second.

> 📂 **Go to:** [orchestrator.py line 143](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L143) — the `await` that paces everything

**Step 2** — The frame gets tossed into the STT's input queue. This is instant — just `queue.put()`.

> 📂 **Go to:** [orchestrator.py line 149](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L149)

**Step 3** — We drain ALL available STT results. The worker might have processed multiple frames since our last drain.

> 📂 **Go to:** [orchestrator.py → _drain_and_process_results()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L194-L217) — the drain loop

**Step 4** — Depends on whether we're in normal mode or flush mode. More on this below.

> 📂 **Go to:** [orchestrator.py lines 158-173](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L158-L173) — the if/else decision

---

## Audio Capture

Microphone recording lives in [audio_io.py → MicrophoneInput](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py#L23-L116).

```
Mic Hardware
    │
    ▼
sounddevice opens an InputStream at 24kHz, mono, blocksize=1920
    │
    │  Every 80ms, sounddevice calls _callback() in a C THREAD
    │
    ▼
_callback() runs in C thread (NOT Python!)
    │  Gets: indata shape (1920, 1) — 1920 samples, 1 channel
    │  Extracts: indata[:, 0].copy() — shape (1920,)
    │
    │  ⚠️ .copy() is CRITICAL! sounddevice reuses the buffer!
    │
    │  Can't touch asyncio directly from C thread, so uses:
    │  loop.call_soon_threadsafe(queue.put_nowait, audio)
    │
    ▼
asyncio.Queue (thread-safe bridge)
    │
    │  Orchestrator calls: await mic.get_frame()
    │
    ▼
Returns numpy array (1920,) float32 = 80ms of audio
```

The tricky part is the C thread → asyncio bridge. sounddevice's callback runs in a C thread that can't use asyncio. So we save the event loop reference during [start()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py#L77-L95) and use `call_soon_threadsafe()` in the [_callback()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py#L48-L73).

> 📂 **Go to:** [audio_io.py → _callback()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py#L48-L73) — where audio enters the system
>
> 📂 **Go to:** [audio_io.py → get_frame()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py#L106-L116) — where the orchestrator receives it

---

## The STT Engine

The STT engine lives in [stt_engine.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L1-L86). It wraps the neural network and runs it in a dedicated thread.

### Why a dedicated thread?

The model uses **streaming contexts** (`mimi.streaming()` and `lm_gen.streaming()`) that hold the KV-cache — the model's memory of everything said so far. These MUST stay open for the entire conversation. You can't open and close them per-frame.

An async function can't hold a context manager open permanently (it would need to yield between frames, breaking the context). So we use a thread that opens the context once and loops forever.

> 📂 **Go to:** [stt_engine.py → _worker()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L313-L390) — the worker loop, the most important function

### Loading the model

[load_model()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L171-L224) does two downloads:

1. **`kyutai/stt-1b-en_fr`** — the main model (transformer + Mimi codec)  
2. **`kyutai/stt-1b-en_fr-candle`** — the extra heads weights

The extra heads aren't in the PyTorch checkpoint — they shipped separately in the Candle format (for Rust). We download that file and attach the weights:

```python
with safe_open(candle_path, framework="pt") as f:
    for i in range(4):
        weight = f.get_tensor(f"extra_heads.{i}.weight")
        head = nn.Linear(2048, 6, bias=False, ...)
        head.weight.data = weight
        lm.extra_heads.append(head)
```

> 📂 **Go to:** [stt_engine.py lines 196-214](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L196-L214) — extra heads loading

---

## The Model

The model has 1 billion parameters and this architecture:

```
Raw Audio (1920 samples, 80ms)
        │
        ▼
┌─────────────────────┐
│     Mimi Codec       │  Neural audio compressor
│     (encoder)        │  1920 raw samples → 20 discrete tokens
│                      │  Like MP3, but neural — learns the best
│                      │  compression for speech
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Transformer       │  Decoder-only, 16 layers
│    (1B parameters)   │  d_model=2048, 16 attention heads
│                      │
│    Maintains KV-     │  Remembers EVERYTHING said so far
│    cache across      │  in the conversation. This is why the
│    frames            │  streaming context must stay open.
└──────────┬──────────┘
           │
       Hidden State
       (2048 dims)
           │
    ┌──────┴──────────────────┐
    │                         │
    ▼                         ▼
text_linear               extra_heads[0-3]
(2048 → 8000)             (2048 → 6 each)
    │                         │
    ▼                         ▼
Token Prediction          State Classification
"What text goes here?"    "What's happening right now?"

  Token 0 = word boundary     head[2], class 0 = PAUSE PROBABILITY
  Token 3 = PAD (nothing)     ~0.001 during speech
  Token 4+ = real text        ~0.9 on genuine pause
```

### Why extra_heads, not text_linear?

This is THE key insight. Both come from the same model, same hidden state. But they answer different questions:

**text_linear** asks: *"What is the next text token?"*
Between words, the answer is PAD (token 3). So P(PAD) spikes to 0.997 on every single word gap. Useless for pause detection.

**extra_heads[2]** asks: *"Is the user done speaking?"*
Between words during a sentence, the answer is NO. The head stays at 0.001. Only when the user genuinely finishes does it rise to 0.9.

> 📂 **Go to:** [stt_engine.py lines 341-356](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L341-L356) — where both outputs are extracted per frame

---

## Word Emission

The model outputs one text token per frame (every 80ms). But tokens ≠ words. A word might be multiple tokens:

```
"thinking" → [token("▁th"), token("ink"), token("ing")]
```

We buffer tokens and emit the word when we see a BOUNDARY. The logic is in the [worker at lines 358-382](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L358-L382):

```
Frame 1: token 365 ("▁So")     → buffer = [365]
Frame 2: token 3   (PAD)       → EMIT "So"! buffer cleared
Frame 3: token 3   (PAD)       → buffer empty, skip
Frame 4: token 142 ("▁I")      → buffer = [142]
Frame 5: token 3   (PAD)       → EMIT "I"! buffer cleared
Frame 6: token 277 ("▁th")     → buffer = [277]
Frame 7: token 489 ("ink")     → buffer = [277, 489]
Frame 8: token 3   (PAD)       → EMIT "think"! buffer cleared
```

Two trigger conditions:
- **Token 0 (word boundary)** — means a new word is starting, so the old word is done
- **Token 3 (PAD)** — means silence; if buffer has tokens, emit them now

We MUST emit on PAD. Without it, the **last word** of an utterance stays stuck in the buffer forever (because no new boundary comes after it).

The emitted word flows through [result_queue](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L383-L390) → [_drain_and_process_results()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L194-L217) → [_on_word()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L218-L252).

> 📂 **Go to:** [stt_engine.py lines 358-382](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L358-L382) — the buffering logic
>
> 📂 **Go to:** [orchestrator.py → _on_word()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L218-L252) — what happens when a word arrives

---

## Pause Detection

This is the most important feature. How does the system know when you've finished speaking?

### The Signal

Every frame, the STT worker extracts `pr_vad = vad_heads[2][0, 0, 0]` — a float between 0 and 1:

| Audio State | `pr_vad` value |
|-------------|---------------|
| Speaking | ~0.001 |
| Between words (mid-sentence) | ~0.001 ← **stays low!** |
| Mid-sentence pause ("I... um...") | ~0.3 |
| End-of-sentence pause | rises to ~0.9 |
| Full silence | ~0.9 |

> 📂 **Go to:** [stt_engine.py lines 341-356](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L341-L356) — where pr_vad is extracted

### The EMA

The raw `pr_vad` goes through an EMA (Exponential Moving Average) to smooth it slightly:

```python
self.ema.update(dt=FRAME_TIME_SEC, new_value=result.pr_vad)
```

With attack=0.01 and release=0.01, the EMA tracks the signal almost instantly. This works because the signal is already smooth (no between-word spikes).

> 📂 **Go to:** [ema.py → update()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/ema.py#L45-L69) — the smoothing math
>
> 📂 **Go to:** [orchestrator.py line 209](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L209) — where EMA is updated each frame

### The EMA Reset Trick

When the first word of a new turn arrives, we reset the EMA to 0.0:

```python
if is_new:
    self.ema.value = 0.0  # ← THE RESET
```

Without this, the EMA would be at ~1.0 from the previous silence. The first word would arrive and P(PAD) would drop to 0.001, but the EMA might not have fallen below 0.6 yet — triggering a false pause!

By resetting to 0.0, we guarantee the EMA starts from scratch every turn.

> 📂 **Go to:** [orchestrator.py lines 246-249](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L246-L249) — the EMA reset

### The Decision

[_determine_pause()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L254-L273) checks two things:

1. **Is the user currently speaking?** → `conversation_state() == "user_speaking"`
2. **Is the EMA above 0.6?** → `self.ema.value > PAUSE_THRESHOLD`

Both must be true. If the state is `"waiting_for_user"` or `"bot_speaking"`, we never trigger.

> 📂 **Go to:** [orchestrator.py → _determine_pause()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L254-L273)

### Timeline of a Real Pause

```
Time  │ What User Does     │ pr_vad │ EMA   │ State            │ Action
──────┼────────────────────┼────────┼───────┼──────────────────┼─────────────
0.0s  │ Silence            │ 0.910  │ 1.00  │ waiting_for_user │ (nothing)
0.5s  │ Says "Hello"       │ 0.003  │ 0.00  │ user_speaking    │ EMA RESET!
0.6s  │ Gap after "Hello"  │ 0.001  │ 0.001 │ user_speaking    │ (stays low)
0.7s  │ Says "world"       │ 0.002  │ 0.002 │ user_speaking    │ (still low)
0.8s  │ Stops speaking     │ 0.001  │ 0.001 │ user_speaking    │ (just stopped)
1.0s  │ Silence            │ 0.400  │ 0.40  │ user_speaking    │ rising...
1.1s  │ Silence            │ 0.700  │ 0.70  │ user_speaking    │ CROSSES 0.6!
      │                    │        │       │                  │ → START FLUSH
1.1s  │ Flush starts       │        │       │                  │ Feed 8 silence frames
1.6s  │ Flush completes    │        │       │                  │ → GENERATE RESPONSE
```

---

## The Flush

When a pause is detected, we can't respond immediately. The STT model has a **0.5s lookahead buffer** — it needs future audio context to predict accurately. So there might be words still in the pipeline.

[_start_flush()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L277-L312) handles this:

```python
# Mark when the flush will be done
self.stt_end_of_flush_time = self.stt.current_time + self.stt.delay_sec

# Feed 8 silence frames to push out any buffered words
silence = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
for _ in range(num_frames):
    self.stt.feed_frame(silence)

# Return immediately — NO BLOCKING!
```

The silence frames go into the STT's input queue. The worker processes them (they push any buffered words through the pipeline). Meanwhile, the main loop continues running — it still feeds mic audio and drains results.

When [stt.current_time > stt_end_of_flush_time](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L170-L173), the flush is done and we call [_generate_response()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L315-L356).

`current_time` is a property that returns `_frame_count × 0.08` — how much audio the worker has actually processed.

> 📂 **Go to:** [orchestrator.py → _start_flush()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L277-L312) — feeding silence, setting timer
>
> 📂 **Go to:** [stt_engine.py → current_time](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L155-L167) — how we track flush progress
>
> 📂 **Go to:** [orchestrator.py → _generate_response()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L315-L356) — what happens after flush

---

## The State Machine

The conversation state is determined by [conversation.py → conversation_state()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/conversation.py#L52-L82). It looks at the LAST message in chat history:

```
┌─────────────────────┐
│  waiting_for_user    │ ← Last msg: {user, ""} or no history
│                      │
│  Pause detection OFF │   _check_silence() active (7s timer)
│  EMA at ~1.0         │
└──────────┬──────────┘
           │
           │ First word arrives → add_message_delta("Hello", "user")
           │ Returns is_new=True → EMA reset to 0.0
           │
           ▼
┌─────────────────────┐
│   user_speaking      │ ← Last msg: {user, "Hello world"}
│                      │
│  Pause detection ON  │   EMA tracking pr_vad
│  Words accumulating  │   EMA > 0.6 → PAUSE!
└──────────┬──────────┘
           │
           │ _start_flush() → _generate_response()
           │ add_message_delta("I heard...", "assistant")
           │
           ▼
┌─────────────────────┐
│   bot_speaking       │ ← Last msg: {assistant, "I heard you say..."}
│                      │
│  (Future: TTS plays) │   (Future: interruption detection)
└──────────┬──────────┘
           │
           │ Bot finishes → add_message_delta("", "user")
           │ Empty message sets state back to waiting_for_user
           │
           ▼
┌─────────────────────┐
│  waiting_for_user    │ ← Back to start! Ready for next turn.
└─────────────────────┘
```

### The Empty Message Trick

This is the cleverest pattern (from Unmute). After the bot responds, we add:

```python
self.conversation.add_message_delta("", "user")
```

This creates `{"role": "user", "content": ""}`. Since content is empty, `conversation_state()` returns `"waiting_for_user"` — pause detection disabled!

When the user says their first word, [add_message_delta("Hello", "user")](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/conversation.py#L84-L128) appends to the empty message: `"" + "Hello" = "Hello"`. It returns `True` because `last_content == ""` (the message WAS empty = this is a new turn).

The orchestrator sees `is_new=True` → resets EMA to 0.0.

> 📂 **Go to:** [conversation.py → add_message_delta()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/conversation.py#L84-L128) — the full logic with comments
>
> 📂 **Go to:** [orchestrator.py line 350](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L347-L355) — where the empty message is added after responding

---

## Silence Detection

If nobody speaks for 7 seconds, [_check_silence()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L358-L381) kicks in:

```python
if elapsed > SILENCE_TIMEOUT:
    self.conversation.add_message_delta("...", "user")
```

Adding `"..."` to the conversation changes state from `"waiting_for_user"` to `"user_speaking"` (because `"..."` is non-empty content). On the next frame, `_determine_pause()` fires (EMA is already ~1.0 since nobody is speaking), and the bot responds to the silence.

> 📂 **Go to:** [orchestrator.py → _check_silence()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L358-L381)

---

## Thread Architecture

Three threads run simultaneously:

```
┌──────────────────────────────────────────────────────────────┐
│                    THREAD 1: sounddevice (C)                  │
│                                                               │
│  Runs in a C thread (not Python!). Fires every 80ms.         │
│  Callback: audio_io.py → _callback() (line 48)              │
│  Puts audio frames into asyncio.Queue via call_soon_threadsafe│
└──────────────────────────┬───────────────────────────────────┘
                           │
                    asyncio.Queue
                    (thread-safe bridge)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    THREAD 2: asyncio main (Python)            │
│                                                               │
│  Runs the orchestrator's main loop.                          │
│  orchestrator.py → run() (line 127)                          │
│  Pops frames from mic, pushes to STT, drains results,       │
│  makes decisions (pause, silence, flush).                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                    threading.Queue (mic_queue)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    THREAD 3: STT worker (Python)              │
│                                                               │
│  Runs GPU inference continuously.                            │
│  stt_engine.py → _worker() (line 313)                        │
│  Blocks on mic_queue.get(), processes frame on GPU (~15ms),  │
│  puts STTResult into result_queue.                           │
└──────────────────────────────────────────────────────────────┘
                           │
                    threading.Queue (result_queue)
                           │
                           ▼
                    (back to Thread 2, drained by
                     _drain_and_process_results)
```

> 📂 **Go to:** [stt_engine.py lines 127-145](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py#L127-L145) — the queue definitions and why each queue type is used

---

## Config

All tunable values live in [config.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/config.py#L1-L80). Every value matches Unmute's production settings.

| Parameter | Value | Used in | Purpose |
|-----------|-------|---------|---------|
| `SAMPLE_RATE` | 24,000 Hz | audio_io.py | Required by Mimi codec |
| `SAMPLES_PER_FRAME` | 1,920 | audio_io.py | 80ms of audio per frame |
| `FRAME_TIME_SEC` | 0.08s | ema, orchestrator | Time per frame |
| `EMA_ATTACK_TIME` | 0.01s | orchestrator | EMA rise speed |
| `EMA_RELEASE_TIME` | 0.01s | orchestrator | EMA fall speed |
| `PAUSE_THRESHOLD` | 0.6 | orchestrator | EMA above this = pause |
| `STT_DELAY_SEC` | 0.5s | stt_engine, orchestrator | Model's lookahead buffer |
| `SILENCE_TIMEOUT` | 7.0s | orchestrator | Seconds before "..." |

> 📂 **Go to:** [config.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/config.py#L1-L80) — full config with comments explaining every value

---

## What's Next

The STT → pause detection pipeline is complete. What's left:

| Phase | What | Status |
|-------|------|--------|
| 5 | **LLM Integration** — Replace `fake_response` with real Ollama/Qwen call | TODO |
| 6 | **TTS Integration** — Stream LLM text through PocketTTS → speaker | TODO |
| 7 | **Interruption** — Clear speaker queue when user talks over bot | TODO |
| 8 | **Fillers** — Play "Hmm", "Uh-huh" while user speaks | TODO |

The response generation happens in [orchestrator.py → _generate_response()](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py#L315-L356). The TODO on line 337 is where the LLM call will go.

---

## File Map (Quick Reference)

| File | Lines | What It Does |
|------|-------|-------------|
| [main.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/main.py) | 39 | Entry point — just starts the orchestrator |
| [core/config.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/config.py) | 80 | All tunable constants |
| [core/audio_io.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/audio_io.py) | 200 | MicrophoneInput + SpeakerOutput |
| [core/conversation.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/conversation.py) | 170 | Chat history + state machine |
| [core/orchestrator.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/core/orchestrator.py) | 381 | Main loop — the brain |
| [stt/stt_engine.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/stt_engine.py) | 390 | STT model + GPU worker thread |
| [stt/ema.py](file:///media/nikki/Data/Projects/AntiGravity/fake-it-till-you-make-it/stt/ema.py) | 69 | EMA filter |
