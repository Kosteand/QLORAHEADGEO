"""
Step 2 sanity check: verify that the RewardModel pools the correct
token position. If this fails, training will silently learn nothing
useful (the head will be reading from pad positions).

The test: run a forward pass on a sequence, then on the same sequence
padded with extra pad tokens. The reward should be IDENTICAL because
the attention mask says the pad tokens don't exist.

Run from project root:
    python scripts/verify_pooling.py
"""

import sys
import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, "/home/claude/rm-project")
from src.model import RewardModel, RewardModelConfig


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
    
    # Forward 1: no padding
    with torch.no_grad():
        out1 = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        r1 = out1.logits.float().item()
    print(f"Reward (no padding): {r1:.6f}")
    
    # Forward 2: pad with 50 extra pad tokens on the right
    pad_id = tok.pad_token_id
    pad_ids = torch.full((1, 50), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros((1, 50), dtype=torch.long, device=device)
    
    ids_padded = torch.cat([enc.input_ids, pad_ids], dim=1)
    mask_padded = torch.cat([enc.attention_mask, pad_mask], dim=1)
    
    with torch.no_grad():
        out2 = model(input_ids=ids_padded, attention_mask=mask_padded)
        r2 = out2.logits.float().item()
    print(f"Reward (right-padded with 50 pad tokens): {r2:.6f}")
    
    # Forward 3: pad with 50 pad tokens on the LEFT
    ids_lpadded = torch.cat([pad_ids, enc.input_ids], dim=1)
    mask_lpadded = torch.cat([pad_mask, enc.attention_mask], dim=1)
    
    with torch.no_grad():
        out3 = model(input_ids=ids_lpadded, attention_mask=mask_lpadded)
        r3 = out3.logits.float().item()
    print(f"Reward (left-padded with 50 pad tokens): {r3:.6f}")
    
    # Compare the pooled hidden states directly, not just the rewards
    import torch
    model.eval()

    with torch.no_grad():
        out1 = model.model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
        out2 = model.model.model(input_ids=ids_padded, attention_mask=mask_padded, return_dict=True)

    from src.model import last_token_pool
    h1 = last_token_pool(out1.last_hidden_state, enc.attention_mask)
    h2 = last_token_pool(out2.last_hidden_state, mask_padded)

    print("h1 shape:", h1.shape, "h2 shape:", h2.shape)
    print("max abs diff in pooled hidden states:", (h1.float() - h2.float()).abs().max().item())
    print("relative diff:", ((h1.float() - h2.float()).norm() / h1.float().norm()).item())
    # Check
    tol = 1e-2  # bf16 has limited precision; this is a reasonable tolerance
    assert abs(r1 - r2) < tol, (
        f"FAIL: right-padding changed reward by {abs(r1-r2):.4f}. "
        f"Pooling is reading from pad positions or attention mask is wrong."
    )
    assert abs(r1 - r3) < tol, (
        f"FAIL: left-padding changed reward by {abs(r1-r3):.4f}. "
        f"Pooling is not handling left-padded sequences correctly."
    )
    print()
    print(f"[OK] Pooling invariant under padding (tolerance {tol})")
    print(f"[OK] Verify pooling check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())