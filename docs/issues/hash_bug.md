# Bug: hash_config inconsistency for first optimizer entry

## Issue

`run.py` mutates `training_params` inside the optimizer loop (adds `mb_subsampling`), but
`hash_config` is called *before* this mutation on the first entry. Result: the first optimizer
entry in any multi-entry config gets a hash that excludes `mb_subsampling`, while all
subsequent entries include it. Hashes are therefore inconsistent across entries.

This was discovered when trying to reconstruct filenames from config files in the notebook
(`load_sweep` in `make_plots_dap_opnorm.ipynb`). The workaround is to mimic run.py's exact
mutation order: compute hash first, then set `training_params['mb_subsampling'] = None`.

## Proposed fix

Freeze a snapshot of `training_params` before the loop for hashing purposes:

```python
training_params_for_hash = copy.deepcopy(training_params)
for opt_config in list_optimizer_params:
    for lr in opt_config['lr']:
        config_hash = hash_config(opt_config_copy, training_params_for_hash, config['gpt_model'])
        ...
        training_params['mb_subsampling'] = None  # still mutate the live copy
```

## Caveat

Fixing this would invalidate existing hashes for all entries 2+ in multi-entry configs,
making old output files unresolvable by the corrected logic. Defer until a good migration
strategy is in place (e.g. rerun affected experiments, or keep both hash schemes).
