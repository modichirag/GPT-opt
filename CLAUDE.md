# GPT-opt Project Conventions

## Directory structure
- `train_scripts/` -- one script per (optimizer, model, dataset). All params use `${N:-default}`.
- `slurm_scripts/` -- submission infrastructure only (submit.sh, sbatch.sh, etc.)
- `param_configs/` -- JSON sweep grids. One per experiment group.
- `hydra_conf/` -- Hydra base configs (model, optimizer, training, data). Don't duplicate.
- `outputs/` -- all experiment results, structured as `outputs/{model}/{dataset}/{optimizer}/`

## Running experiments
- Train scripts: `train_scripts/{optimizer}_{model}_{dataset}.sh <lr> <wd> ...`
- Sweeps: `bash slurm_scripts/submit.sh train_scripts/<script>.sh param_configs/<sweep>.json <group_name> <gpus>`
- **Smoke tests must finish in under 2 minutes.** The goal is to verify the code runs — nothing more. Minimize ALL work: training steps, validation, compilation, logging. If a smoke test takes longer than 2 minutes, something is wrong — investigate and fix it, don't just wait. Use ALL of these flags:
  - `WANDB_MODE=offline` — no W&B uploads
  - `+training.training_params.max_steps=1` — single step is enough to verify code runs
  - `training.training_params.val_tokens_processed=262144` — reduce validation (~256K tokens instead of 8M)
  - `training.training_params.compile=False` — skip torch.compile if on 80GB GPU (only needed on 40GB to avoid OOM)

## Adding new experiments
- New model/dataset/optimizer combo -> new train script in `train_scripts/`
- New hyperparameter sweep -> new JSON in `param_configs/`
- Do NOT create new train scripts for hyperparameter variants (use param_configs JSON instead)
