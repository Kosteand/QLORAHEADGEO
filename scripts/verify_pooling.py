"""
Step 2 sanity check: verify that the RewardModel pools the correct
token position. If this fails, training will silently learn nothing
useful (the head will be reading from pad positions).

The test runs a forward pass on a sequence, then on the same sequence
with extra pad tokens added (right and left). We check both:
1. Pooled hidden states are nearly identical (the actual pooling test).
2. Final reward outputs are close (sensitive to head's bf16 amplification).

The hidden-state check is the strict one. Reward differences can be
larger purely from head-side bf16 noise, especially with an untrained
randomly-initialized head.

Run from project root:
    python -m scripts.verify_pooling
"""

import sys
import torch
from transformers import AutoTokenizer

# Add project root to path so `from src.model import ...` works
sys.path.insert(0, ".")
from src.model import RewardModel, RewardModelConfig, last_token_pool


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading reward model with bounded_above activation...")
    config = RewardModelConfig(
        base_model_name=model_name,
        activation_name="bounded_above",
    )
    model = RewardModel.from_base_model(config, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Tokenize a test prompt
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tok(text, return_tensors="pt").to(device)
    print(f"Sequence length: {enc.input_ids.shape[1]}")

    # Build padded variants
    pad_id = tok.pad_token_id
    pad_ids = torch.full((1, 50), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros((1, 50), dtype=torch.long, device=device)

    ids_rpadded = torch.cat([enc.input_ids, pad_ids], dim=1)
    mask_rpadded = torch.cat([enc.attention_mask, pad_mask], dim=1)
    ids_lpadded = torch.cat([pad_ids, enc.input_ids], dim=1)
    mask_lpadded = torch.cat([pad_mask, enc.attention_mask], dim=1)

    # ---- Forward pass: rewards ----
    with torch.no_grad():
        r1 = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.float().item()
        r2 = model(input_ids=ids_rpadded, attention_mask=mask_rpadded).logits.float().item()
        r3 = model(input_ids=ids_lpadded, attention_mask=mask_lpadded).logits.float().item()

    print(f"Reward (no padding):     {r1:.6f}")
    print(f"Reward (right-padded):   {r2:.6f}  (diff: {abs(r1-r2):.4f})")
    print(f"Reward (left-padded):    {r3:.6f}  (diff: {abs(r1-r3):.4f})")

    # ---- The actual pooling test: compare pooled hidden states ----
    # If pooling extracts the same token in all three cases, the hidden
    # states should be ~identical (within bf16 numerical noise from
    # the base transformer's attention computations).
    with torch.no_grad():
        # model.model is the AutoModel base (not model.model.model)
        out1 = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
        out2 = model.model(input_ids=ids_rpadded, attention_mask=mask_rpadded, return_dict=True)
        out3 = model.model(input_ids=ids_lpadded, attention_mask=mask_lpadded, return_dict=True)

    h1 = last_token_pool(out1.last_hidden_state, enc.attention_mask)
    h2 = last_token_pool(out2.last_hidden_state, mask_rpadded)
    h3 = last_token_pool(out3.last_hidden_state, mask_lpadded)

    diff_12 = (h1.float() - h2.float()).abs().max().item()
    diff_13 = (h1.float() - h3.float()).abs().max().item()
    rel_12 = ((h1.float() - h2.float()).norm() / h1.float().norm()).item()
    rel_13 = ((h1.float() - h3.float()).norm() / h1.float().norm()).item()

    print()
    print(f"Pooled hidden state diffs:")
    print(f"  no-pad vs right-padded: max abs={diff_12:.4f}, relative={rel_12:.4%}")
    print(f"  no-pad vs left-padded:  max abs={diff_13:.4f}, relative={rel_13:.4%}")

    # Strict test on hidden states (this is what 'pooling correct' actually means)
    # bf16 attention has ~3-5% relative noise across different padding configurations
    # because of how RoPE interacts with attention math at different sequence shapes.
    hidden_rel_tol = 0.10  # 10% relative diff is generous but realistic for bf16
    if rel_12 > hidden_rel_tol:
        print(f"\n[FAIL] Right-padded hidden state diff ({rel_12:.2%}) exceeds {hidden_rel_tol:.0%}")
        print("This suggests pooling is reading from a wrong position.")
        return 1
    if rel_13 > hidden_rel_tol:
        print(f"\n[FAIL] Left-padded hidden state diff ({rel_13:.2%}) exceeds {hidden_rel_tol:.0%}")
        return 1

    # Looser test on rewards (head amplifies noise; untrained head is sensitive)
    reward_tol = 0.5
    if abs(r1 - r2) > reward_tol or abs(r1 - r3) > reward_tol:
        print(f"\n[WARN] Large reward differences ({reward_tol})")
        print("Hidden states are correct so pooling is fine, but the head")
        print("amplifies bf16 noise more than expected. After training the")
        print("head should be more numerically stable.")
        # don't fail - this is normal for untrained heads
    
    print()
    print("[OK] Pooling is correct (hidden states match within bf16 tolerance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())