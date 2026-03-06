# disBatch Sweep Pipeline

## Overview

The standard way to run hyperparameter sweeps is:

```
slurm_scripts/submit.sh <train_script.sh> <params.json> <group_name> <num_gpus>
```

This:
1. Copies `train_script.sh` → `logs/<group_name>/run_info_N/train.sh`
2. Copies `params.json` → `logs/<group_name>/run_info_N/params.json`
3. Calls `generate_task_file.py` to expand all parameter combinations into a `tasks` file
4. Submits `slurm_scripts/sbatch.sh` which runs `disBatch tasks` across `<num_gpus>` tasks

## Writing a train script

The bash script uses positional parameters with defaults:

```bash
#!/bin/bash
lr=${1:-0.02}
damping=${2:-0.0}

python run_hydra.py \
    model=gpt-tiny \
    training=finewebmini_tiny \
    data=finewebmini \
    optimizer=dap-opnorm \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.damping=${damping}"
```

`generate_task_file.py` parses the `${N:-default}` pattern to discover parameter names and positions.

## Writing a params.json

```json
{
    "lr": [0.02, 0.03, 0.05, 0.07, 0.1],
    "damping": [0.03]
}
```

All combinations are generated (Cartesian product). Single-element lists fix a parameter.

## Example invocation

```bash
slurm_scripts/submit.sh slurm_scripts/train_dap_opnorm.sh \
    param_configs/dap_opnorm_lr_sweep.json \
    dap_opnorm_damping003_lr \
    6
```

This requests 6 GPUs and runs 6 tasks (one per LR value) in parallel via disBatch.

## Where to put files

- Training scripts: `slurm_scripts/train_<experiment>.sh`
- Param configs: `param_configs/<experiment>.json`
- Logs land in: `logs/<group_name>/run_info_N/`

## Notes

- `run_info` directories are auto-incremented by `slurm_scripts/rename_utils.sh`
- Each task's stdout/stderr goes to `logs/<group_name>/run_info_N/logs/log_XX.out`
- The `project_config.sh` sets `ROOT_DIR` to the repo root
- `slurm_scripts/sbatch.sh` requests `--gpus-per-task=1 --cpus-per-task=4`; Slurm sets `CUDA_VISIBLE_DEVICES` automatically per task
- The venv is activated by `sbatch.sh` before running disBatch — train scripts do NOT need `source venv/activate`
- **GPU constraint**: `sbatch.sh` uses `--constraint=a100` to avoid landing on Blackwell nodes (workergpu171-181, sm_120) which are not supported by the current PyTorch install (max sm_90). Do not remove this constraint.
- **Task count**: Pass the desired parallelism as `<num_gpus>` to `submit.sh` — this equals the number of tasks to run concurrently. Tasks will span multiple nodes as needed.
