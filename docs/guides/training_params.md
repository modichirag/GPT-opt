# Training Parameter Terminology

This document clarifies the naming conventions used in training configs and experiment logs,
since some parameter names are potentially confusing.

## Parameter definitions

| Parameter | Meaning | Example (`finewebmini_tiny`) |
|-----------|---------|------------------------------|
| `tokens_processed` | **Tokens per optimizer step** (= effective global batch size in tokens) | 262,144 = 2^18 |
| `batch_size` | Micro-batch size: sequences per GPU per forward pass | 8 |
| `context_length` | Tokens per sequence | 1,024 |
| `grad_accum_steps` | Micro-steps accumulated before each optimizer step = `tokens_processed / (world_size × batch_size × context_length)` | 262144 / (1×8×1024) = **32** |
| `global_batch_size` | Sequences per optimizer step = `tokens_processed / context_length` | 256 |
| Total optimizer steps | `total_dataset_tokens / tokens_processed` per epoch | ~381 for finewebmini (~100M tokens) |

## Key point: `tokens_processed` is *not* the training budget

Despite the name, `tokens_processed` is **not** the total number of tokens seen during training.
It is the number of tokens consumed **per optimizer step** — i.e., the global batch size expressed
in tokens.

From `gptopt/train_distributed.py`:

```python
grad_accum_steps = tokens_processed // (world_size * batch_size * context_length)
total_optimizer_steps = total_tokens_in_dataset // tokens_processed  # per epoch
```

The total training budget is determined by `total_optimizer_steps × tokens_processed`, which
equals `total_tokens_in_dataset × num_epochs`.

## Batch size in experiment docs

When experiment logs or docs say **"batch size 256"**, this refers to `global_batch_size`
(sequences per optimizer step), **not** the micro-batch `batch_size` (sequences per GPU per
forward pass).

## Example: finewebmini_tiny vs fineweb1B_tiny

| Setting | `tokens_processed` | `batch_size` | `global_batch_size` | Optimizer steps (1 epoch) |
|---------|-------------------|--------------|---------------------|--------------------------|
| `finewebmini_tiny` | 262,144 | 8 | 256 | ~381 (~100M tokens) |
| `fineweb1B_tiny`   | 262,144 | 8 | 256 | ~3,815 (~1B tokens)  |

Both use the same per-step batch size; `fineweb1B_tiny` just runs ~10× more steps.
