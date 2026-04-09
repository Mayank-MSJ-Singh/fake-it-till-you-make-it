"""Test: Does the PyTorch model (kyutai/stt-1b-en_fr) already have extra heads?"""
import torch
import moshi.models

print("Loading PyTorch model (kyutai/stt-1b-en_fr)...")
info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)

print(f"\nlm.extra_heads type: {type(lm.extra_heads)}")
print(f"lm.extra_heads length: {len(lm.extra_heads)}")
print(f"lm.extra_heads contents: {lm.extra_heads}")

if len(lm.extra_heads) > 0:
    print("\n✅ Extra heads ARE included in the PyTorch model!")
    for i, head in enumerate(lm.extra_heads):
        print(f"  head[{i}]: {head}")
        print(f"    weight shape: {head.weight.shape}")
        print(f"    weight device: {head.weight.device}")
        print(f"    weight dtype: {head.weight.dtype}")
        # Check if weights are all zeros (placeholder) or real
        w = head.weight.data.float()
        print(f"    weight norm: {w.norm().item():.4f}  (0 = placeholder, >0 = real)")
else:
    print("\n❌ Extra heads are EMPTY in the PyTorch model.")
    print("   We DO need the candle checkpoint.")
