# ============================================================
# 🎛️  ALL TUNABLE VALUES — change here, reflects everywhere
# ============================================================

# Audio
SAMPLE_RATE = 24_000              # Hz — required by Mimi codec
SAMPLES_PER_FRAME = 1_920         # 80ms per frame
FRAME_TIME_SEC = 0.08             # 1920 / 24000

# EMA smoothing for pause detection
# Signal: P(PAD) from text logits — naturally semantic:
#   - During speech: P(PAD) ≈ 0.0
#   - Mid-sentence pause: P(PAD) ≈ 0.7 (model knows sentence is incomplete)
#   - End-of-sentence: P(PAD) ≈ 0.999 (model knows speech is done)
EMA_ATTACK_TIME = 0.15            # Smooths over 1-frame between-word gaps
EMA_RELEASE_TIME = 0.05           # Fast drop when user resumes speaking
EMA_INITIAL_VALUE = 1.0           # Start as "paused"

# Pause detection
# Threshold 0.85 means mid-sentence pauses (P≈0.7 → EMA≈0.7) won't trigger,
# but end-of-sentence (P≈0.999 → EMA crosses 0.85 in ~480ms) will.
# No MIN_SPEAKING_TIME needed — the EMA reset to 0.0 on new message start
# naturally prevents premature triggers (takes ~480ms to rise to 0.85).
PAUSE_THRESHOLD = 0.85

# STT
STT_WARMUP_FRAMES = 12            # Skip first 12 frames (~960ms) — VAD is noisy during warmup
STT_DELAY_SEC = 0.5               # Inherent STT buffering delay (same as Unmute)
FLUSH_FRAMES = 8                  # ceil(0.5 / 0.08) + 1 — silence frames to flush STT

# Silence
SILENCE_TIMEOUT = 7.0             # Seconds of silence before bot responds with "..."

# Interruption (Phase 7)
INTERRUPTION_VAD_THRESHOLD = 0.4  # Below this → user started speaking during bot
UNINTERRUPTIBLE_TIME = 3.0        # Seconds grace period after bot starts

# Fillers (Phase 8)
FILLER_EMA_LOW = 0.3              # Below = user actively speaking
FILLER_EMA_HIGH = 0.55            # Above = probably end of sentence
FILLER_COOLDOWN = 6.0             # Seconds between fillers
FILLER_MIN_SPEAKING_TIME = 3.0    # Don't filler too early
FILLER_VOLUME = 0.7               # 70% of normal volume