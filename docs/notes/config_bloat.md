# Issue: Config bloat in configs/finewebmini/ (RESOLVED)

## Problem

Each sweep experiment requires a new yaml file under `configs/finewebmini/`, leading to
proliferation (14+ files already for the dap_opnorm experiments alone). This makes it hard
to track what was run and why.

## Root cause

`run.py` takes a single config file and loops over all optimizer entries in it. There is no
way to override individual hyperparams from the CLI -- every new sweep needs a new file.

## Resolution

Migrated to `run_hydra.py` + `hydra_conf/` with CLI overrides. Sweep variation now lives in
`param_configs/` JSON files, not in duplicated train scripts or config yamls.

See `CLAUDE.md` at repo root for current conventions.
