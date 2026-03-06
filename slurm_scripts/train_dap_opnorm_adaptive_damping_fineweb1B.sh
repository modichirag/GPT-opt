#!/bin/bash
lr=${1:-0.03}
opnorm_target=${2:-2.0}
ema_beta=${3:-0.99}

python -u run_hydra.py \
    model=gpt-tiny \
    training=fineweb1B_tiny \
    data=fineweb1B \
    optimizer=dap-opnorm \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.opnorm_target=${opnorm_target}" \
    "optimizer.optimizer_params.ema_beta=${ema_beta}"
