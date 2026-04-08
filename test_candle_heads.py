"""Check if the Candle model has extra head weights in its safetensors."""
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

# Download the candle model's safetensors
print("Downloading candle model info...")
path = hf_hub_download("kyutai/stt-1b-en_fr-candle", "model.safetensors")

print(f"File: {path}")
print("\nLooking for extra_head weights...")

with safe_open(path, framework="pt") as f:
    all_keys = f.keys()
    
    # Find any keys related to extra heads
    head_keys = [k for k in all_keys if "extra" in k.lower() or "head" in k.lower()]
    
    if head_keys:
        print(f"\n✅ Found {len(head_keys)} extra head weights:")
        for k in head_keys:
            tensor = f.get_tensor(k)
            print(f"  {k}: shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print("\n❌ No extra head weights found")
        print("\nAll weight keys containing interesting terms:")
        for k in sorted(all_keys):
            if any(term in k.lower() for term in ["linear", "out", "proj"]):
                tensor = f.get_tensor(k)
                print(f"  {k}: {tensor.shape}")

    print(f"\nTotal keys: {len(all_keys)}")
