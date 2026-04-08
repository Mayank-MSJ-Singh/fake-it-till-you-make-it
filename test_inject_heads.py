"""Test: Load extra head weights from Candle model into PyTorch model."""
import torch
import torch.nn as nn
import moshi.models
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# 1. Load the PyTorch STT model (no extra heads)
print("Loading PyTorch model...")
info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
mimi = info.get_mimi(device="cuda")
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)

print(f"Before: extra_heads = {lm.extra_heads}")  # Empty ModuleList

# 2. Load extra head weights from Candle model
print("\nLoading extra head weights from Candle model...")
candle_path = hf_hub_download("kyutai/stt-1b-en_fr-candle", "model.safetensors")

with safe_open(candle_path, framework="pt") as f:
    for i in range(4):
        weight = f.get_tensor(f"extra_heads.{i}.weight")
        # Create a Linear layer (in=2048, out=6, no bias)
        head = nn.Linear(2048, 6, bias=False, dtype=torch.bfloat16, device="cuda")
        head.weight.data = weight.to(device="cuda", dtype=torch.bfloat16)
        lm.extra_heads.append(head)

print(f"After: extra_heads = {lm.extra_heads}")

# 3. Test it with silence — does step_with_extra_heads now return heads?
print("\nTesting with audio frames...")
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
silence = torch.zeros((1, 1, 1920), device="cuda")

with mimi.streaming(1), lm_gen.streaming(1):
    for i in range(20):
        tokens = mimi.encode(silence)
        text_tokens, vad_heads = lm_gen.step_with_extra_heads(tokens)
        
        if i == 15:
            print(f"\nFrame {i}:")
            print(f"  vad_heads length: {len(vad_heads)}")
            for j, h in enumerate(vad_heads):
                val = h[0, 0, 0].item()
                print(f"  vad_heads[{j}]: shape={h.shape}, value={val:.4f}")

print("\n✅ Done!")
