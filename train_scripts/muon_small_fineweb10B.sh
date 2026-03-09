#!/bin/bash
lr=${1:-0.02}
wd=${2:-0.1}
max_steps=${3:-0}
seed=${4:-42}

python -u run_hydra.py \
    model=gpt-small \
    training=fineweb10B \
    data=fineweb10B \
    optimizer=muon \
    logging=wandb \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "+training.training_params.max_steps=${max_steps}" \
    "seed=${seed}"
