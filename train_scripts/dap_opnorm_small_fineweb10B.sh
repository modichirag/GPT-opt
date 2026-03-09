#!/bin/bash
lr=${1:-0.03}
wd=${2:-0.1}
damping=${3:-0.0}
opnorm_target=${4:-2.0}
per_layer_damping=${5:-true}
ema_beta=${6:-0.99}
output_cov_mode=${7:-null}
max_steps=${8:-0}
seed=${9:-42}
schedule_steps=${10:-0}

python -u run_hydra.py \
    model=gpt-small \
    training=fineweb10B \
    data=fineweb10B \
    optimizer=dap-opnorm \
    paths=dap-opnorm-sweep \
    logging=wandb \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.damping=${damping}" \
    "optimizer.optimizer_params.opnorm_target=${opnorm_target}" \
    "optimizer.optimizer_params.per_layer_damping=${per_layer_damping}" \
    "optimizer.optimizer_params.ema_beta=${ema_beta}" \
    "optimizer.optimizer_params.output_cov_mode=${output_cov_mode}" \
    "+training.training_params.max_steps=${max_steps}" \
    "+training.training_params.schedule_steps=${schedule_steps}" \
    "seed=${seed}"
