#!/bin/bash
lr=${1:-0.03}
wd=${2:-0.1}
damping=${3:-0.0}
opnorm_target=${4:-2.0}
per_layer_damping=${5:-true}
ema_beta=${6:-0.99}
seed=${7:-42}

python -u run_hydra.py \
    model=gpt-small \
    training=fineweb1B \
    data=fineweb1B \
    optimizer=dap-opnorm \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.damping=${damping}" \
    "optimizer.optimizer_params.opnorm_target=${opnorm_target}" \
    "optimizer.optimizer_params.per_layer_damping=${per_layer_damping}" \
    "optimizer.optimizer_params.ema_beta=${ema_beta}" \
    "seed=${seed}"
