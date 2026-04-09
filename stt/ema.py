"""
Exponential Moving Average (EMA)
=================================

Smooths a noisy signal over time. Used to smooth the raw pause
probability from the STT model's extra heads.

The key feature: ASYMMETRIC smoothing.
  - "attack" = signal going UP   (user stops speaking → pause rising)
  - "release" = signal going DOWN (user starts speaking → pause falling)

You can make it rise slowly but fall quickly, or vice versa.
In our case both are 0.01 (nearly instant) because the extra heads
signal is already smooth and doesn't need much filtering.

Math:
  alpha = 1 - exp(-dt × ln(2) / half_life)
  new_ema = old_ema + alpha × (new_value - old_ema)

  With half_life=0.01 and dt=0.08:
    alpha = 1 - exp(-0.08 × 0.693 / 0.01) = 0.996
    → Each frame, EMA moves 99.6% toward the new value.

This is a direct copy of Unmute's implementation:
  unmute/stt/exponential_moving_average.py
"""
import math


class ExponentialMovingAverage:
    """Smooths a noisy signal using exponential moving average.

    Attributes:
        attack_time:  Half-life in seconds when signal is RISING.
        release_time: Half-life in seconds when signal is FALLING.
        value:        Current smoothed value. Can be read anytime,
                      and manually set (e.g., reset to 0.0 on new turn).
    """

    def __init__(self, attack_time: float, release_time: float, initial_value: float = 1.0):
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = initial_value

    def update(self, dt: float, new_value: float):
        """Update the EMA with a new sample.

        Args:
            dt: Time elapsed since last update (seconds). For us, always 0.08.
            new_value: The new raw value to blend toward.
        """
        # Pick the appropriate half-life based on direction
        if new_value > self.value:
            # Signal is RISING (attack) — user might be stopping
            half_life = self.attack_time
        else:
            # Signal is FALLING (release) — user might be starting to speak
            half_life = self.release_time

        # Compute the blending factor (alpha)
        # Higher alpha = faster tracking of new value
        if half_life > 0:
            alpha = 1 - math.exp(-dt * math.log(2) / half_life)
        else:
            # half_life == 0 means instant tracking (no smoothing)
            alpha = 1.0

        # Blend: move `value` toward `new_value` by `alpha` fraction
        self.value += alpha * (new_value - self.value)
