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
- Smoke tests: prefix with `WANDB_MODE=offline`, use `+training.training_params.max_steps=10`

## Adding new experiments
- New model/dataset/optimizer combo -> new train script in `train_scripts/`
- New hyperparameter sweep -> new JSON in `param_configs/`
- Do NOT create new train scripts for hyperparameter variants (use param_configs JSON instead)
