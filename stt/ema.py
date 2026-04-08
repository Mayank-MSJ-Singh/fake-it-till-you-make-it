import math


class ExponentialMovingAverage:
    """Smooths a noisy signal using exponential moving average.
    
    attack = signal going UP (speaking → paused)
    release = signal going DOWN (paused → speaking)
    """

    def __init__(self, attack_time: float, release_time: float, initial_value: float = 1.0):
        self.attack_time = attack_time
        self.release_time = release_time
        self.value = initial_value

    def update(self, dt: float, new_value: float):
        """Update the EMA with a new sample."""
        if new_value > self.value:
            half_life = self.attack_time
        else:
            half_life = self.release_time

        if half_life > 0:
            alpha = 1 - math.exp(-dt * math.log(2) / half_life)
        else:
            alpha = 1.0

        self.value += alpha * (new_value - self.value)
