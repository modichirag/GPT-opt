#!/bin/bash
lr=${1:-0.02}
wd=${2:-0.1}
seed=${3:-42}

python -u run_hydra.py \
    model=gpt-small \
    training=fineweb1B \
    data=fineweb1B \
    optimizer=muon \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "seed=${seed}"
