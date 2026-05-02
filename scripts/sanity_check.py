"""
Step 1 sanity check: verify Qwen2.5-0.5B-Instruct loads and generates
coherent text. If this fails, nothing else will work.

Run from project root:
    python scripts/sanity_check.py
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"Model loaded. Hidden size: {model.config.hidden_size}, "
          f"num layers: {model.config.num_hidden_layers}")
    print(f"Tokenizer pad_token: {tok.pad_token!r}, "
          f"pad_token_id: {tok.pad_token_id}")
    print(f"Tokenizer eos_token: {tok.eos_token!r}, "
          f"eos_token_id: {tok.eos_token_id}")
    
    # Verify chat template applies correctly
    messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print()
    print("Chat-template-formatted prompt:")
    print(repr(formatted))
    
    # Generate
    inputs = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    
    response = tok.decode(out[0], skip_special_tokens=False)
    print()
    print("Generated response:")
    print(response)
    print()
    
    # Verify the response contains the key chat tokens
    assert "<|im_start|>" in response, "Chat template tokens missing!"
    assert "<|im_end|>" in response, "Chat template end tokens missing!"
    print("[OK] Chat template tokens present in output")
    print("[OK] Sanity check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())