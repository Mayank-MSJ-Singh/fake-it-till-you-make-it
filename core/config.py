"""
Configuration — All Tunable Values
====================================

Every magic number in the project lives here. If you want to change
how the system behaves, this is the ONLY file you need to touch.

These values match Unmute's production settings exactly.
"""

# ============================================================
# 🔊  AUDIO
# ============================================================
# The Mimi audio codec (from Kyutai) requires exactly 24kHz audio.
# sounddevice records at this rate, and every frame is 1920 samples
# which equals exactly 80 milliseconds of audio.
SAMPLE_RATE = 24_000              # Hz — required by Mimi codec
SAMPLES_PER_FRAME = 1_920         # 80ms per frame (24000 × 0.08)
FRAME_TIME_SEC = 0.08             # Seconds per frame (1920 / 24000)

# ============================================================
# 📊  EMA (Exponential Moving Average) for Pause Detection
# ============================================================
# The raw pause signal comes from extra_heads[2][0] — one of the
# model's classification heads. This signal is already smooth:
#   - During speech (even between words): ~0.001
#   - End-of-sentence silence: rises to ~0.9
#
# Because the signal is already clean, we barely need to smooth it.
# attack_time = release_time = 0.01 means the EMA tracks the raw
# signal almost instantly (99.6% of new value per frame).
#
# Matches Unmute exactly: speech_to_text.py lines 87-89
EMA_ATTACK_TIME = 0.01            # Half-life for RISING signal (speaking → paused)
EMA_RELEASE_TIME = 0.01           # Half-life for FALLING signal (paused → speaking)
EMA_INITIAL_VALUE = 1.0           # Start as "paused" (user hasn't spoken yet)

# ============================================================
# ⏸️  PAUSE DETECTION
# ============================================================
# When the EMA crosses this threshold, we consider the user done speaking.
# With extra_heads signal:
#   - Between words: signal ≈ 0.001 → EMA ≈ 0.001 → no trigger ✅
#   - Real pause:    signal ≈ 0.9   → EMA ≈ 0.9   → crosses 0.6 → PAUSE ✅
#
# Matches Unmute: unmute_handler.py line 387
PAUSE_THRESHOLD = 0.6

# ============================================================
# 🧠  STT (Speech-to-Text) MODEL
# ============================================================
# The STT model has a built-in 0.5s delay — it needs to "look ahead"
# at future audio to make accurate predictions. This means when the
# user stops speaking, there's 0.5s of audio still buffered in the
# model that hasn't been transcribed yet.
#
# After detecting a pause, we feed silence frames to flush this buffer
# and push out any remaining words.
STT_WARMUP_FRAMES = 12            # Skip first 12 frames (~960ms) — model output is noisy
STT_DELAY_SEC = 0.5               # Model's lookahead buffer (same as Unmute)
FLUSH_FRAMES = 8                  # ceil(0.5 / 0.08) + 1 — silence frames to flush

# ============================================================
# 🤫  SILENCE DETECTION
# ============================================================
# If the user doesn't speak for this long after the bot finishes,
# inject "..." into the conversation. This triggers a bot response
# to fill the awkward silence (like "Are you still there?").
SILENCE_TIMEOUT = 7.0             # Seconds of silence before bot responds

# ============================================================
# 🛑  INTERRUPTION (Phase 7 — not yet implemented)
# ============================================================
# When the bot is speaking and the user starts talking over it,
# these values control when to cut the bot off.
INTERRUPTION_VAD_THRESHOLD = 0.4  # EMA below this = user is speaking
UNINTERRUPTIBLE_TIME = 3.0        # Grace period at conversation start (echo issues)

# ============================================================
# 💬  FILLERS (Phase 8 — not yet implemented)
# ============================================================
# "Hmm", "Uh-huh" sounds to play while the user is speaking,
# to make the bot feel more human.
FILLER_EMA_LOW = 0.3              # Below = user actively speaking
FILLER_EMA_HIGH = 0.55            # Above = probably end of sentence
FILLER_COOLDOWN = 6.0             # Seconds between fillers
FILLER_MIN_SPEAKING_TIME = 3.0    # Don't filler too early in the turn
FILLER_VOLUME = 0.7               # 70% of normal volume