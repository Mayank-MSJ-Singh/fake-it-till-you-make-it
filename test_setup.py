# Test 1: Can sounddevice see your mic and speakers?
import sounddevice as sd
print("=== Audio Devices ===")
print(sd.query_devices())
print(f"\nDefault input:  {sd.default.device[0]}")
print(f"Default output: {sd.default.device[1]}")

# Test 2: Can moshi load?
import moshi.models
print("\n=== Moshi ===")
print("moshi loaded successfully!")
