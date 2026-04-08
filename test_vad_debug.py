import moshi.models

info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
print("stt_config:", info.stt_config)
print()
print("raw_config:", info.raw_config)
print()

# Also check the LM model itself for extra heads
import torch
lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
print("\nlm.extra_heads:", getattr(lm, 'extra_heads', 'NOT FOUND'))
print("lm.n_extra_heads:", getattr(lm, 'n_extra_heads', 'NOT FOUND'))

# Check all attributes with 'head' or 'extra' in the name
for attr in dir(lm):
    if 'head' in attr.lower() or 'extra' in attr.lower():
        print(f"lm.{attr}: {getattr(lm, attr, 'N/A')}")
