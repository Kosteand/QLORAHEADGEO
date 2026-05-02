"""
Step 4 sanity check: verify the pipeline can OVERFIT on 1K examples.

If this fails (training accuracy stays < 90% on 1K examples after 3 epochs),
the pipeline is broken. Common causes:
  1. LoRA not attached to the head (check modules_to_save).
  2. Pooling at the wrong token position (run verify_pooling.py first).
  3. pad_token_id not set on model config.
  4. Chat template not applied (check inspect_example output).
  5. Head initialization too large (gradients vanish through saturated phi).

This script trains on 1000 UltraFeedback pairs for 3 epochs and asserts
that final training accuracy exceeds OVERFIT_THRESHOLD. Total runtime
on a 0.5B model with an A10/A100 should be 5-10 minutes.

We test the LINEAR head (phi=identity) here. The linear head is the
easiest to overfit and the cleanest sanity check. If this passes for
phi=identity, the pipeline is correct, and any failure-to-overfit
with concave phi is about the head, not the pipeline.

Run from project root:
    python scripts/overfit_test.py
"""

import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_ultrafeedback, inspect_example
from src.model import RewardModel, RewardModelConfig

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

train_ds, _ = load_ultrafeedback(tok, max_length=1024, n_train=2)
ex = train_ds[0]
print("CHOSEN:", tok.decode(ex["input_ids_chosen"])[-300:])
print()
print("REJECTED:", tok.decode(ex["input_ids_rejected"])[-300:])

OVERFIT_THRESHOLD = 0.90  # require 90%+ training accuracy by end


def main():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    activation_name = "ident"  # linear baseline - easiest to overfit
    n_train = 1000
    n_eval = 100
    output_dir = "./outputs/overfit-test"
    
    print("=" * 70)
    print("OVERFIT TEST")
    print("=" * 70)
    print(f"  Model: {base_model_name}")
    print(f"  activation: {activation_name}")
    print(f"  n_train: {n_train}, n_eval: {n_eval}")
    print(f"  Threshold: training accuracy must reach {OVERFIT_THRESHOLD:.0%}")
    print()
    
    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---- Data ----
    print("Loading data...")
    train_ds, eval_ds = load_ultrafeedback(
        tokenizer, max_length=1024, n_train=n_train, n_eval=n_eval
    )
    print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")
    
    # Inspect one example to confirm tokenization is sane
    print()
    print("Inspecting first training example (look for chat template tokens):")
    inspect_example(train_ds[0], tokenizer)
    print()
    
    # ---- Model ----
    print("Building model...")
    rm_config = RewardModelConfig(
        base_model_name=base_model_name,
        activation_name=activation_name,
    )
    model = RewardModel.from_base_model(rm_config, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # LoRA - small ranks are fine for overfit test
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,  # NO dropout for overfit test - we WANT to memorize
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["reward_head"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Verify head is trainable
    head_params = [
        (n, p) for n, p in model.named_parameters()
        if "reward_head" in n and p.requires_grad
    ]
    if not head_params:
        print()
        print("FAIL: reward_head has NO trainable parameters!")
        print("Check modules_to_save in the LoRA config.")
        return 1
    print(f"[OK] reward_head trainable params:")
    for n, p in head_params:
        print(f"    {n}: shape {tuple(p.shape)}")
    
    model.enable_input_require_grads()
    
    # ---- Train ----
    # 3 epochs, no eval, very small batch, higher LR than usual.
    # We're trying to memorize 1000 examples - everything is tuned for that.
    args = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-4,
        lr_scheduler_type="constant",  # no decay - we want to keep pushing
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        max_length=1024,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    
    trainer = RewardTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=train_ds.select(range(min(50, len(train_ds)))),  # tiny placeholder
    )
    
    print()
    print("Training (3 epochs on 1000 examples)...")
    train_result = trainer.train()
    print()
    
    # ---- Evaluate on training set ----
    # We want to know: did we memorize the training data?
    # Run trainer.evaluate on the TRAINING set to get pairwise accuracy.
    print("Evaluating on TRAINING set (overfit check)...")
    train_metrics = trainer.evaluate(eval_dataset=train_ds)
    train_accuracy = train_metrics.get("eval_accuracy", None)
    
    if train_accuracy is None:
        print(f"WARNING: 'eval_accuracy' not in metrics. Available: {list(train_metrics.keys())}")
        # Try alternate keys
        for key in ["accuracy", "eval_pairwise_accuracy", "pairwise_accuracy"]:
            if key in train_metrics:
                train_accuracy = train_metrics[key]
                print(f"  Using '{key}' = {train_accuracy:.4f}")
                break
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training loss (final): {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}" if train_accuracy is not None
          else "Training accuracy: COULD NOT EXTRACT")
    print()
    
    if train_accuracy is not None and train_accuracy >= OVERFIT_THRESHOLD:
        print(f"[OK] PASS: training accuracy {train_accuracy:.4f} >= {OVERFIT_THRESHOLD}")
        print("Pipeline is working correctly.")
        return 0
    else:
        print(f"[FAIL] training accuracy {train_accuracy} < {OVERFIT_THRESHOLD}")
        print()
        print("Likely causes (in order of frequency):")
        print("  1. LoRA modules_to_save not including reward_head")
        print("     -> check 'reward_head trainable params' output above")
        print("  2. Pooling at wrong position (run verify_pooling.py first)")
        print("  3. pad_token_id not set on model config")
        print("  4. Chat template not applied (check inspect_example output)")
        print("  5. enable_input_require_grads not called")
        return 1


if __name__ == "__main__":
    sys.exit(main())