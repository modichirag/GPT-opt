#!/bin/bash
lr=${1:-0.02}
damping=${2:-0.0}
ema_beta=${3:-0.0}
wd=${4:-0.1}
seed=${5:-42}

python run_hydra.py \
    model=gpt-tiny \
    training=finewebmini_tiny \
    data=finewebmini \
    optimizer=dap-opnorm \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.damping=${damping}" \
    "optimizer.optimizer_params.ema_beta=${ema_beta}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "seed=${seed}"
