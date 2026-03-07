#!/bin/bash
lr=${1:-0.003}
wd=${2:-0.1}
seed=${3:-42}

python -u run_hydra.py \
    model=gpt-small \
    training=fineweb1B \
    data=fineweb1B \
    optimizer=adamw \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.lr_schedule=warm-up-cosine" \
    "optimizer.optimizer_params.warm_up_fraction=0.1" \
    "seed=${seed}"
