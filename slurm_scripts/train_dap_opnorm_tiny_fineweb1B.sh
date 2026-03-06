#!/bin/bash
lr=${1:-0.05}
damping=${2:-0.001}
ema_beta=${3:-0.99}
wd=${4:-0.1}
seed=${5:-42}

python -u run_hydra.py \
    model=gpt-tiny \
    training=fineweb1B_tiny \
    data=fineweb1B \
    optimizer=dap-opnorm \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.damping=${damping}" \
    "optimizer.optimizer_params.ema_beta=${ema_beta}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "seed=${seed}"
