"""Test that we can load the Kyutai STT model."""
import torch
import moshi.models

print("Loading STT model (this will download ~2GB on first run)...")

info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")

print("Loading Mimi codec...")
mimi = info.get_mimi(device="cuda")
print(f"  Mimi sample rate: {mimi.sample_rate}")
print(f"  Mimi frame rate: {mimi.frame_rate}")
print(f"  Mimi frame size: {mimi.frame_size}")

print("Loading STT transformer...")
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)

print("Creating LMGen...")
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

print("Loading tokenizer...")
tokenizer = info.get_text_tokenizer()

print("\n✅ All models loaded successfully!")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
