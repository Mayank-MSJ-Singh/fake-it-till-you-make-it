# Understanding Pause Probability in Unmute

> A deep-dive into how Unmute decides when a user has finished speaking and it's time for the AI to respond.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Where Does the Raw Pause Signal Come From?](#where-does-the-raw-pause-signal-come-from)
3. [The `prs` Array — What Each Index Means](#the-prs-array--what-each-index-means)
4. [Exponential Moving Average (EMA) Smoothing](#exponential-moving-average-ema-smoothing)
5. [The `determine_pause()` Decision Function](#the-determine_pause-decision-function)
6. [What Happens After a Pause Is Detected?](#what-happens-after-a-pause-is-detected)
7. [Pause Prediction Also Powers Interruption Detection](#pause-prediction-also-powers-interruption-detection)
8. [Edge Case Handling](#edge-case-handling)
9. [Key Constants & Tuning Knobs](#key-constants--tuning-knobs)
10. [Full Data Flow Diagram](#full-data-flow-diagram)
11. [File-by-File Reference](#file-by-file-reference)
12. [Background: DSM-ASR & the "Action Stream"](#background-dsm-asr--the-action-stream)

---

## The Big Picture

Unmute is a voice conversation system: a user **speaks**, their speech is transcribed in real-time, an LLM generates a text response, and a TTS model reads it aloud. The critical challenge is: **when exactly should the system stop listening and start responding?**

Too early → the bot cuts the user off mid-sentence.  
Too late → the conversation feels sluggish and unnatural.

The **pause probability** system solves this. It is a real-time signal, produced every 80ms by the STT (speech-to-text) model, that estimates:  
> *"Has the user finished their thought, or are they just pausing mid-sentence?"*

This is often referred to as **Semantic VAD** (Voice Activity Detection) — because unlike traditional VAD which only detects silence vs. sound, this system understands the *semantic context* of speech to make its prediction.

---

## Where Does the Raw Pause Signal Come From?

The STT server is built on **DSM-ASR** (Delayed Streams Modeling for Automatic Speech Recognition), developed by [Kyutai](https://kyutai.org/). The [pre-print paper](https://arxiv.org/abs/2509.08753) describes this architecture.

### How DSM-ASR works (simplified):

1. User audio is encoded by the **Mimi codec** into discrete tokens at **12.5 Hz** (one frame every 80ms).
2. A decoder-only Transformer processes audio tokens as input and predicts text tokens as output — both aligned to the same 12.5 Hz time grid.
3. At each time step, the model predicts **what comes next in the text stream**. There are special tokens:
   - **`PAD`** — "No word here" (silence / continuation of the current word)
   - **`WORD`** — "A new word starts here"
   - **Regular text tokens** — The actual word content

4. **The model outputs probabilities** over these tokens at every step. These probabilities become the **`prs` (probabilities) array** sent to the Python backend.

The STT server sends a `Step` message every 80ms (one frame) that contains:

```python
class STTStepMessage(BaseModel):
    type: Literal["Step"]
    step_idx: int
    prs: list[float]  # Probabilities from the model
```

### What is a "Semantic VAD"?

Traditional VAD: "Is there acoustic energy? → Someone is speaking."  
Semantic VAD: "Given what the user just said, is their sentence **semantically complete**?"

For example, if a user says *"I went to the..."* and pauses — a traditional VAD might trigger (silence detected!), but the semantic VAD knows the sentence is incomplete and **waits**.

If the user says *"I went to the store."* and pauses — the semantic VAD detects the sentence is semantically complete and **triggers the bot's response**.

---

## The `prs` Array — What Each Index Means

The `prs` array contains probabilities from the DSM-ASR model. Based on how it's used in the code:

```python
# From speech_to_text.py, line 212-213
self.pause_prediction.update(
    dt=FRAME_TIME_SEC, new_value=message.prs[2]
)
```

**`prs[2]`** is specifically used as the pause prediction signal. In the context of the DSM-ASR model's text-stream prediction:

| Index | Likely Meaning | Description |
|-------|---------------|-------------|
| `prs[0]` | Probability of `PAD` | "No new word at this time step" (user still mid-word or silence) |
| `prs[1]` | Probability of `WORD` | "A new word starts at this time step" (user is actively speaking) |
| `prs[2]` | **Probability of end-of-utterance / pause** | "The user has finished speaking" — **this is the pause probability** |

The model has been trained on aligned speech-text pairs, so it learns from millions of examples what a "finished thought" sounds like vs. a "mid-sentence pause." This is what makes it a **semantic** VAD — it uses the full context of the transcribed text and audio, not just acoustic features.

---

## Exponential Moving Average (EMA) Smoothing

The raw `prs[2]` value from the model would be noisy if used directly — it could spike up and down frame-to-frame. To smooth this, Unmute applies an **Exponential Moving Average (EMA)** with asymmetric attack/release times.

### The EMA class:

```python
# From unmute/stt/exponential_moving_average.py

class ExponentialMovingAverage:
    def __init__(self, attack_time: float, release_time: float, initial_value: float = 0.0):
        """
        attack_time:  Time in seconds to reach 50% of a RISING target value.
        release_time: Time in seconds to decay to 50% of a FALLING target value.
        initial_value: Starting value of the EMA.
        """
```

### How the math works:

```
alpha = 1 - exp(-dt / time_constant * ln(2))
value = (1 - alpha) * old_value + alpha * new_value
```

Where:
- `dt` = time between updates (0.08 seconds = `FRAME_TIME_SEC`)
- `time_constant` = either `attack_time` or `release_time`, depending on direction
- The `ln(2)` factor ensures the `time_constant` parameter represents the **half-life** (time to reach 50% of the target)

### In Unmute's configuration:

```python
# From speech_to_text.py, lines 87-89
self.pause_prediction = ExponentialMovingAverage(
    attack_time=0.01,    # Very fast attack  (speaking → not speaking)
    release_time=0.01,   # Very fast release (not speaking → speaking)
    initial_value=1.0    # Start as "user is NOT speaking" (high = paused)
)
```

**Important terminology (from the code comments):**
- **Attack** = "from speaking to not speaking" → the pause prediction value RISES
- **Release** = "from not speaking to speaking" → the pause prediction value FALLS

With `attack_time = 0.01` and `release_time = 0.01`, both directions respond very quickly. Given `dt = 0.08s`:

```
alpha = 1 - exp(-0.08 / 0.01 * ln(2))
alpha = 1 - exp(-5.545)
alpha ≈ 0.996
```

This means **each new frame almost completely replaces the old value** — the EMA is very responsive with very little smoothing. This is intentional: the DSM-ASR model already provides a relatively stable signal, so only minimal smoothing is needed.

### The initial value of 1.0:

The EMA starts at `1.0`, meaning "user is not speaking." This prevents the system from trying to detect pauses before the user has started talking.

### Warm-up period:

```python
# From speech_to_text.py, lines 183-184, 209-214
# The pause prediction is all over the place in the first few steps, so ignore.
n_steps_to_wait = 12

if n_steps_to_wait > 0:
    n_steps_to_wait -= 1
else:
    self.pause_prediction.update(
        dt=FRAME_TIME_SEC, new_value=message.prs[2]
    )
```

The first **12 steps** (~960ms) of STT output are discarded because the model's predictions are unreliable during startup. Only after this warm-up does the EMA begin tracking the real pause probability.

---

## The `determine_pause()` Decision Function

This is the core decision point. Located in `unmute_handler.py`:

```python
def determine_pause(self) -> bool:
    stt = self.stt
    if stt is None:
        return False
    if self.chatbot.conversation_state() != "user_speaking":
        return False

    # Wall clock time since last ASR word was received
    time_since_last_message = (
        stt.sent_samples / self.input_sample_rate
    ) - self.stt_last_message_time

    if stt.pause_prediction.value > 0.6:
        return True
    else:
        return False
```

### The conditions:

1. **STT must be connected** — `stt is not None`
2. **User must be currently speaking** — `conversation_state() == "user_speaking"`
3. **Pause probability must exceed 0.6** — `stt.pause_prediction.value > 0.6`

### Understanding `conversation_state()`:

The conversation has three states, determined by the chat history:

```python
class Chatbot:
    def conversation_state(self) -> ConversationState:
        last_message = self.chat_history[-1]
        if last_message["role"] == "assistant":
            return "bot_speaking"
        elif last_message["role"] == "user":
            if last_message["content"].strip() != "":
                return "user_speaking"
            else:
                return "waiting_for_user"
        elif last_message["role"] == "system":
            return "waiting_for_user"
```

| State | Meaning | Pause detection? |
|-------|---------|-----------------|
| `"waiting_for_user"` | Bot finished, waiting for user to start | ❌ No |
| `"user_speaking"` | User is currently saying something | ✅ Yes |
| `"bot_speaking"` | Bot is generating/speaking a response | ❌ No (but **interruption** detection is active) |

### The 0.6 threshold:

This is the key tuning parameter. The pause probability is a float between 0.0 and 1.0:

- **< 0.4** → User is clearly still speaking (also used for interruption detection)
- **0.4 – 0.6** → Uncertain zone, no action taken
- **> 0.6** → User has likely finished → **trigger response**

This hysteresis band (0.4 to 0.6) prevents rapid oscillation between states.

---

## What Happens After a Pause Is Detected?

When `determine_pause()` returns `True`, a specific "flushing" pipeline kicks in:

```python
# From unmute_handler.py, lines 340-370 (the receive() method)
if self.determine_pause():
    logger.info("Pause detected")

    # 1. Notify the frontend
    await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

    # 2. Set up the flush window
    self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
    self.stt_flush_timer = Stopwatch()

    # 3. Send silence frames to flush the STT pipeline
    num_frames = int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
    zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
    for _ in range(num_frames):
        await stt.send_audio(zero)
```

### Step-by-step:

#### Step 1: Notify the Frontend
An `InputAudioBufferSpeechStopped` event is sent so the UI can update (e.g., change a "listening" indicator).

#### Step 2: Start the Flush Window
Because DSM-ASR uses a **delay** between audio and text streams (default `STT_DELAY_SEC = 0.5`), the STT model might still be processing audio that was spoken *before* the pause. We need to let it finish processing that backlog.

`stt_end_of_flush_time` marks when we expect the flush to be complete.

#### Step 3: Send Silence to Flush
We feed **zero (silence) frames** into the STT to push out any remaining buffered transcription. The number of silence frames is calculated from the delay:

```python
# 0.5 seconds / 0.08 seconds per frame = 6.25, ceil → 7, +1 = 8 frames
num_frames = int(math.ceil(0.5 / 0.08)) + 1
```

This ensures all buffered audio in the STT pipeline gets processed and any final words are transcribed.

#### Step 4: After Flush Completes — Generate Response

```python
# From unmute_handler.py, lines 363-370
# During the flush period, we check if STT has caught up
if stt.current_time > self.stt_end_of_flush_time:
    self.stt_end_of_flush_time = None
    elapsed = self.stt_flush_timer.time()
    rtf = stt.delay_sec / elapsed
    logger.info("Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf)
    await self._generate_response()
```

Once the STT has processed all the silence, `_generate_response()` is called:
1. The LLM receives the full chat history and generates a text response
2. As the LLM generates tokens, they are streamed to the TTS
3. The TTS produces audio that is sent back to the user

---

## Pause Prediction Also Powers Interruption Detection

The same pause probability signal is also used to detect when the user wants to **interrupt the bot** while it is speaking:

```python
# From unmute_handler.py, lines 352-358
elif (
    self.chatbot.conversation_state() == "bot_speaking"
    and stt.pause_prediction.value < 0.4   # User IS speaking
    and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC  # > 3 seconds
):
    logger.info("Interruption by STT-VAD")
    await self.interrupt_bot()
    await self.add_chat_message_delta("", "user")
```

### The interruption conditions:

1. **Bot is currently speaking** — `conversation_state() == "bot_speaking"`
2. **Pause prediction < 0.4** — The model is confident the user is speaking (low pause probability = user is talking)
3. **More than 3 seconds have elapsed** — `UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3` — this grace period prevents false interruptions caused by echo cancellation issues at the start of the conversation

### There are TWO interruption mechanisms:

| Mechanism | Trigger | Sensitivity |
|-----------|---------|-------------|
| **STT-VAD Interruption** | `pause_prediction < 0.4` during `bot_speaking` | Based on acoustic/semantic analysis |
| **STT-Word Interruption** | A transcribed **word** arrives during `bot_speaking` | More definitive — actual text was recognized |

The word-based interruption (in `_stt_loop()`) is the stronger signal:

```python
# From unmute_handler.py, lines 456-458
if self.chatbot.conversation_state() == "bot_speaking":
    logger.info("STT-based interruption")
    await self.interrupt_bot()
```

---

## Edge Case Handling

### 1. First Word Reset

When the STT sends the **first word** of a new user utterance, the pause prediction is manually reset:

```python
# From unmute_handler.py, lines 462-465
is_new_message = await self.add_chat_message_delta(data.text, "user")
if is_new_message:
    # Ensure we don't stop after the first word if the VAD didn't have
    # time to react.
    stt.pause_prediction.value = 0.0
```

Without this, the system might detect a "pause" right after the first word because the EMA hasn't had time to settle from its initial value of 1.0 (or from the previous pause).

### 2. Empty STT Messages Ignored

```python
# From unmute_handler.py, lines 450-454
# The STT sends an empty string as the first message, but we
# don't want to add that because it can trigger a pause even
# if the user hasn't started speaking yet.
if data.text == "":
    continue
```

### 3. Long Silence Detection

If the user doesn't speak for 7 seconds after the bot finishes, a special "..." message is injected:

```python
# From unmute_handler.py, lines 626-638
USER_SILENCE_TIMEOUT = 7.0

async def detect_long_silence(self):
    if (
        self.chatbot.conversation_state() == "waiting_for_user"
        and (self.audio_received_sec() - self.waiting_for_user_start_time)
        > USER_SILENCE_TIMEOUT
    ):
        logger.info("Long silence detected.")
        await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")
```

This changes the conversation state to `"user_speaking"`, which will then trigger `determine_pause()` on the next frame (and the pause prediction will be high since there's no speech → the bot responds to the silence).

### 4. No Interruption Detection During Flush

```python
# From unmute_handler.py, lines 361-362
# We do not try to detect interruption here, the STT would be processing
# a chunk full of 0, so there is little chance the pause score would 
# indicate an interruption.
```

During the flushing period (when silence is being fed to the STT), the system doesn't try to detect interruptions because the model is processing artificial silence, making its predictions meaningless.

### 5. Dummy STT for Testing

There's a `DummySpeechToText` class that keeps `pause_prediction` permanently at `1.0` (always paused) and never sends any words. This is useful for testing TTS latency in isolation:

```python
# From dummy_speech_to_text.py
# We just keep this at 1.0 = user is not speaking
self.pause_prediction = ExponentialMovingAverage(
    attack_time=0.01, release_time=0.01, initial_value=1.0
)
```

---

## Key Constants & Tuning Knobs

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| `FRAME_TIME_SEC` | 0.08s (80ms) | `kyutai_constants.py` | Time between STT model steps. One audio frame = 1920 samples at 24kHz |
| `SAMPLE_RATE` | 24,000 Hz | `kyutai_constants.py` | Audio sample rate |
| `SAMPLES_PER_FRAME` | 1,920 | `kyutai_constants.py` | Samples per 80ms frame (24000 × 0.08) |
| `STT_DELAY_SEC` | 0.5s | `kyutai_constants.py` | Built-in delay of the DSM-ASR model (text is delayed behind audio) |
| `attack_time` | 0.01s | `speech_to_text.py` | EMA half-life for rising values (very fast) |
| `release_time` | 0.01s | `speech_to_text.py` | EMA half-life for falling values (very fast) |
| `initial_value` | 1.0 | `speech_to_text.py` | Start as "not speaking" |
| `n_steps_to_wait` | 12 | `speech_to_text.py` | Warm-up steps (~960ms) before EMA tracking begins |
| Pause threshold | **0.6** | `unmute_handler.py` | Above this → user finished speaking |
| Interruption threshold | **0.4** | `unmute_handler.py` | Below this during bot_speaking → user is interrupting |
| `UNINTERRUPTIBLE_BY_VAD_TIME_SEC` | 3s | `unmute_handler.py` | Grace period at conversation start (echo cancellation) |
| `USER_SILENCE_TIMEOUT` | 7s | `unmute_handler.py` | Seconds of user silence before bot responds with a prompt |

---

## Full Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER'S BROWSER                         │
│  [Microphone] → [Opus Encoder] → WebSocket → Backend           │
└────────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (unmute_handler.py)                │
│                                                                 │
│  receive(frame):                                                │
│    │                                                            │
│    ├─ self.n_samples_received += len(audio)                     │
│    │                                                            │
│    ├─ Store debug data:                                         │
│    │    { pause_prediction, amplitude, time }                   │
│    │                                                            │
│    ├─ await stt.send_audio(audio)  ──────────────────┐          │
│    │                                                  │          │
│    │              ┌───────────────────────────────────┘          │
│    │              ▼                                              │
│    │  ┌────────────────────────────────┐                        │
│    │  │    STT SERVER (DSM-ASR)        │                        │
│    │  │                                │                        │
│    │  │  Audio → Mimi Codec → Tokens   │                        │
│    │  │  Transformer predicts text     │                        │
│    │  │  stream probabilities          │                        │
│    │  │                                │                        │
│    │  │  Sends back per frame:         │                        │
│    │  │  ┌─────────────────────┐       │                        │
│    │  │  │ STTStepMessage      │       │                        │
│    │  │  │  step_idx: int      │       │                        │
│    │  │  │  prs: [p0, p1, p2]  │───────│───┐                   │
│    │  │  └─────────────────────┘       │   │                    │
│    │  │                                │   │                    │
│    │  │  Also sends:                   │   │                    │
│    │  │  STTWordMessage (when a word   │   │                    │
│    │  │  is transcribed)               │   │                    │
│    │  └────────────────────────────────┘   │                    │
│    │                                       │                    │
│    │                                       ▼                    │
│    │  ┌──────────────────────────────────────────────┐          │
│    │  │  __aiter__() in speech_to_text.py            │          │
│    │  │                                              │          │
│    │  │  case STTStepMessage():                      │          │
│    │  │    if n_steps_to_wait > 0:                   │          │
│    │  │      n_steps_to_wait -= 1  # Skip first 12   │          │
│    │  │    else:                                     │          │
│    │  │      self.pause_prediction.update(            │          │
│    │  │        dt=0.08,           # FRAME_TIME_SEC   │          │
│    │  │        new_value=prs[2]   # The pause prob   │          │
│    │  │      )                                       │          │
│    │  └──────────────────────────────────────────────┘          │
│    │                                                            │
│    ├─ if stt_end_of_flush_time is None:  (normal mode)          │
│    │    │                                                       │
│    │    ├─ detect_long_silence()                                │
│    │    │                                                       │
│    │    ├─ determine_pause():                                   │
│    │    │    state == "user_speaking"?                           │
│    │    │    pause_prediction.value > 0.6?                      │
│    │    │      │                                                │
│    │    │      ├─ YES → PAUSE DETECTED!                         │
│    │    │      │    → Send silence frames to flush STT          │
│    │    │      │    → Set stt_end_of_flush_time                 │
│    │    │      │    → Eventually → _generate_response()         │
│    │    │      │                                                │
│    │    │      └─ NO → Continue listening                       │
│    │    │                                                       │
│    │    └─ Check interruption:                                  │
│    │         state == "bot_speaking"?                            │
│    │         pause_prediction.value < 0.4?                      │
│    │         time > 3 seconds?                                  │
│    │           → YES → interrupt_bot()                          │
│    │                                                            │
│    └─ else:  (flush mode — waiting for STT pipeline to drain)   │
│         if stt.current_time > stt_end_of_flush_time:            │
│           → Flushing done → _generate_response()                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Reference

### `unmute/stt/exponential_moving_average.py`
The EMA filter class. Simple but crucial — smooths the raw model output into a stable signal. Key method: `update(dt, new_value)` which applies the asymmetric smoothing.

### `unmute/stt/speech_to_text.py`
The STT client. Connects to the DSM-ASR server via WebSocket, sends audio, and receives transcription + step messages. The `__aiter__` method processes incoming messages and updates `self.pause_prediction` using `prs[2]` from each `STTStepMessage`.

### `unmute/stt/dummy_speech_to_text.py`  
A testing stub that keeps `pause_prediction` at 1.0 permanently. Used for benchmarking TTS latency without a real STT server.

### `unmute/unmute_handler.py`
The main orchestrator. Contains `determine_pause()` (the decision function), `receive()` (the audio processing loop), and all the logic for flushing, interruption, and response generation.

### `unmute/kyutai_constants.py`
Defines `FRAME_TIME_SEC` (0.08s), `STT_DELAY_SEC` (0.5s), `SAMPLE_RATE` (24kHz), and `SAMPLES_PER_FRAME` (1920).

### `unmute/llm/chatbot.py`
Manages conversation state (`waiting_for_user`, `user_speaking`, `bot_speaking`) based on the chat history. The `determine_pause()` function only fires when state is `"user_speaking"`.

### `unmute/openai_realtime_api_events.py`
Defines the WebSocket protocol events, including `InputAudioBufferSpeechStopped` ("A pause was detected by the VAD") and `UnmuteInterruptedByVAD`.

---

## Background: DSM-ASR & the "Action Stream"

From the [Kyutai paper](https://arxiv.org/abs/2509.08753):

> *"Unmute's speech-to-text is streaming, accurate, and includes a semantic VAD that predicts whether you've actually finished speaking or if you're just pausing mid-sentence, meaning it's low-latency but doesn't interrupt you."*

The DSM (Delayed Streams Modeling) framework works by:

1. **Time-aligning** audio and text to the same frame rate (12.5 Hz via Mimi codec)
2. Using a **decoder-only Transformer** that processes both streams simultaneously  
3. Introducing a **delay** between streams — for ASR, the text stream is delayed behind the audio, so the model has future audio context when predicting what text goes "here"
4. The model predicts the text token at each time step, with probabilities over `PAD`, `WORD`, and regular tokens

The **"action stream"** concept from the TTS side also exists in the ASR side as a byproduct: the model's confidence about whether a word boundary or pause exists at a given time step directly gives us the pause probability signal.

This is fundamentally different from traditional energy-based VAD:
- **Energy-based VAD**: "Is the signal amplitude above a threshold?" → Fails on background noise, breathing, and mid-sentence pauses
- **DSM-ASR Semantic VAD**: "Given the full transcription context so far and the acoustic features, is this a natural completion point?" → Handles mid-sentence pauses gracefully

### The delay trade-off

The STT model uses a 0.5-second delay (`STT_DELAY_SEC`), meaning it "sees" 0.5 seconds of future audio when making predictions. This gives it enough context to make good pause predictions but adds 0.5 seconds to the response latency. After a pause is detected, the system needs to flush this 0.5-second buffer of silence through the pipeline before the final transcription is complete and the LLM can begin generating a response.

---

## Summary: The Complete Timeline

Here's what happens in a typical turn:

| Time | Event |
|------|-------|
| `T+0.0s` | User starts speaking. First word arrives → `pause_prediction` reset to 0.0 |
| `T+0.0s` to `T+N.0s` | User speaks. `prs[2]` stays low, EMA stays below 0.6 |
| `T+N.0s` | User finishes speaking. `prs[2]` starts rising |
| `T+N.08s` | EMA updates, crosses 0.6 threshold |
| `T+N.08s` | `determine_pause()` returns `True` |
| `T+N.08s` to `T+N.58s` | Flushing: ~8 silence frames sent to STT, any remaining words transcribed |
| `T+N.58s` | Flushing complete → `_generate_response()` called |
| `T+N.58s+` | LLM generates text → streamed to TTS → audio sent to user |

**Total turn-taking latency contribution from pause detection: ~80ms** (one frame for detection) + **~500ms** (flushing the STT delay) = **~580ms** before the LLM starts generating. This is on top of the LLM and TTS latency.
