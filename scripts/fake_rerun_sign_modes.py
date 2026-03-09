#!/usr/bin/env python3
"""
Fake-rerun sign mode output JSONs with corrected hyperparams.

Sign modes (sign_input, sign_only) in Exp 1 had opnorm_target scaling the
update redundantly with lr: p -= lr * sign(...) * opnorm_target. The code has
been fixed. This script creates new output files as if we'd run with the
correct config: lr=effective_lr (= old_lr * opnorm_target), opnorm_target=null.

Old files are NOT deleted.
"""

import json
import hashlib
import glob
import os
import shutil
from pathlib import Path

# --- Constants ---

BASE_DIR = Path("outputs/gpt-small/fineweb1B/dap-opnorm")
SOURCE_PATTERN = str(BASE_DIR / "bs-512-lr-0.02-wd-0.1-opnorm-*" / "*.json")
SIGN_MODES = {"sign_input", "sign_only"}
MIN_STEPS = 1000

TRAINING_PARAMS = {
    "tokens_processed": 524288,
    "val_tokens_processed": 8388608,
    "batch_size": 32,
    "num_epochs": 1,
    "context_length": 1024,
    "gradnorm": 1.0,
    "tensorcore_precision": "highest",
    "autocast": True,
    "mixed_precision": "bfloat16",
    "compile": True,
    "mb_subsampling": None,
    "global_batch_size": 512,
    "schedule_steps": 1906,
}

MODEL_CONFIG = {
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "vocab_size": 50257,
    "flash_attention": True,
}


def hash_config(optimizer_config, training_params, gpt_model):
    """Reproduce hash_config from gptopt/utils.py."""
    relevant_fields = {
        "optimizer_config": optimizer_config,
        "training_params": training_params,
        "gpt_model": gpt_model,
    }
    config_str = json.dumps(relevant_fields, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def main():
    source_files = sorted(glob.glob(SOURCE_PATTERN))
    print(f"Found {len(source_files)} JSON files total\n")

    results = []

    for src_path in source_files:
        with open(src_path) as f:
            data = json.load(f)

        hp = data.get("hyperparams", {})
        mode = hp.get("output_cov_mode")
        num_steps = len(data.get("losses", []))
        opnorm_target = hp.get("opnorm_target")

        # Filter: sign modes only, >= MIN_STEPS, valid opnorm_target
        if mode not in SIGN_MODES:
            continue
        if num_steps < MIN_STEPS:
            continue
        if opnorm_target is None or opnorm_target == 0:
            continue

        old_lr = hp["lr"]
        eff_lr = old_lr * opnorm_target

        # Build new opt_config (what hash_config sees)
        new_opt_config = dict(hp)
        new_opt_config["lr"] = eff_lr
        new_opt_config["opnorm_target"] = None

        # Compute new hash
        new_hash = hash_config(new_opt_config, TRAINING_PARAMS, MODEL_CONFIG)

        # Build new filename and directory
        new_dir = BASE_DIR / f"bs-512-lr-{eff_lr}-wd-0.1-opnorm-None"
        new_filename = f"dap-opnorm-lr-{eff_lr}-warm-up-cosine-{new_hash}-world1.json"
        new_path = new_dir / new_filename

        # Build new JSON data
        new_data = dict(data)
        new_hp = dict(hp)
        new_hp["lr"] = eff_lr
        new_hp["opnorm_target"] = None
        new_data["hyperparams"] = new_hp
        new_data["name"] = f"dap-opnorm-lr-{eff_lr}"

        # Write
        os.makedirs(new_dir, exist_ok=True)
        with open(new_path, "w") as f:
            json.dump(new_data, f, indent=2)

        val_losses = data.get("val_losses", [])
        last_val = f"{val_losses[-1]:.3f}" if val_losses else "N/A"

        results.append({
            "mode": mode,
            "old_target": opnorm_target,
            "old_lr": old_lr,
            "eff_lr": eff_lr,
            "val": last_val,
            "old_path": src_path,
            "new_path": str(new_path),
        })

    # Print summary
    print(f"Created {len(results)} fake-rerun JSONs:\n")
    print(f"{'mode':12s} {'target':>6s} {'old_lr':>6s} {'eff_lr':>6s} {'val':>6s}  new_path")
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x["mode"], x["eff_lr"])):
        print(f"{r['mode']:12s} {r['old_target']:6.1f} {r['old_lr']:6.3f} {r['eff_lr']:6.2f} {r['val']:>6s}  {r['new_path']}")

    # Verification: spot-check
    if results:
        print("\n--- Spot check (first result) ---")
        r = results[0]
        with open(r["new_path"]) as f:
            check = json.load(f)
        print(f"  hyperparams.lr = {check['hyperparams']['lr']}")
        print(f"  hyperparams.opnorm_target = {check['hyperparams']['opnorm_target']}")
        print(f"  name = {check['name']}")
        print(f"  num losses = {len(check['losses'])}")
        print(f"  last val_loss = {check['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
