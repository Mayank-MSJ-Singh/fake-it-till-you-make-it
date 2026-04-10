"""Test: Can we load everything from kyutai/stt-1b-en_fr-candle only?

The candle repo has:
  - config.json with extra_heads_num_heads=4, extra_heads_dim=6
  - model.safetensors with extra_heads.*.weight keys
  - Same Mimi codec + tokenizer

If this works, we don't need 2 downloads — just one repo has everything.
"""
import torch
import moshi.models

print("=" * 60)
print("TEST 1: Load model from candle repo ONLY")
print("=" * 60)

print("\n1. Loading CheckpointInfo from kyutai/stt-1b-en_fr-candle...")
info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr-candle")

print(f"\n2. Config loaded. Checking for extra_heads in config:")
print(f"   lm_config keys: {list(info.lm_config.keys()) if info.lm_config else 'None'}")
if info.lm_config:
    print(f"   extra_heads_num_heads: {info.lm_config.get('extra_heads_num_heads', 'NOT IN CONFIG')}")
    print(f"   extra_heads_dim: {info.lm_config.get('extra_heads_dim', 'NOT IN CONFIG')}")

print("\n3. Loading Mimi codec...")
mimi = info.get_mimi(device="cuda")
print(f"   ✅ Mimi loaded")

print("\n4. Loading LM model (this is the big one)...")
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)

print(f"\n5. Checking extra_heads:")
print(f"   type: {type(lm.extra_heads)}")
print(f"   length: {len(lm.extra_heads)}")
if len(lm.extra_heads) > 0:
    print(f"   ✅ Extra heads ARE included!")
    for i, head in enumerate(lm.extra_heads):
        w = head.weight.data.float()
        print(f"   head[{i}]: shape={head.weight.shape}, device={head.weight.device}, norm={w.norm().item():.4f}")
else:
    print(f"   ❌ Extra heads are EMPTY")

print("\n6. Loading tokenizer...")
tokenizer = info.get_text_tokenizer()
print(f"   ✅ Tokenizer loaded, vocab size: {tokenizer.get_piece_size()}")

print("\n7. Creating LMGen and testing step_with_extra_heads...")
lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
silence = torch.zeros((1, 1, 1920), device="cuda")

with mimi.streaming(1), lm_gen.streaming(1):
    for i in range(15):
        tokens = mimi.encode(silence)
        result = lm_gen.step_with_extra_heads(tokens)
        if result is not None and i >= 12:
            text_tokens, vad_heads = result
            print(f"\n   Frame {i}:")
            print(f"   text_tokens shape: {text_tokens.shape}")
            print(f"   text_token value: {text_tokens[0, 0, 0].item()}")
            print(f"   vad_heads count: {len(vad_heads)}")
            for j, h in enumerate(vad_heads):
                vals = h[0, 0, :].float().cpu().tolist()
                print(f"   head[{j}]: {[f'{v:.4f}' for v in vals]}")
            # The pause signal
            pr_vad = vad_heads[2][0, 0, 0].float().item()
            print(f"\n   ✅ pr_vad (prs[2]) = {pr_vad:.4f}")
            print(f"      (should be ~0.3-0.5 during silence)")
            break

print("\n" + "=" * 60)
print("RESULT: Single candle repo works!" if len(lm.extra_heads) > 0 else "RESULT: FAILED")
print("=" * 60)
print(f"\nGPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
