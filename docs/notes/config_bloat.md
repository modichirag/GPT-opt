# Issue: Config bloat in configs/finewebmini/

## Problem

Each sweep experiment requires a new yaml file under `configs/finewebmini/`, leading to
proliferation (14+ files already for the dap_opnorm experiments alone). This makes it hard
to track what was run and why.

## Root cause

`run.py` takes a single config file and loops over all optimizer entries in it. There is no
way to override individual hyperparams from the CLI — every new sweep needs a new file.

## Existing solution (unused)

`run_hydra.py` + `hydra_conf/` already exists and supports CLI overrides, e.g.:

```bash
python run_hydra.py optimizer=dap-opnorm optimizer.optimizer_params.damping=0.05 \
                    optimizer.optimizer_params.lr=0.05
```

No new config file needed per run. Sweeps can be expressed as Hydra multirun:

```bash
python run_hydra.py -m optimizer.optimizer_params.damping=0.01,0.05,0.1 \
                       optimizer.optimizer_params.lr=0.03,0.05,0.07
```

## What's needed

- Add `hydra_conf/optimizer/dap-opnorm.yaml` mirroring the dap-opnorm entry in `utils.py`
- Migrate future dap-opnorm sweeps to use `run_hydra.py` instead of `run.py`
- Update `submit.sh` (or add `submit_hydra.sh`) to wrap Hydra multirun in sbatch
