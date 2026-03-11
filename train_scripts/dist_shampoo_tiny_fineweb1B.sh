#!/bin/bash
lr=${1:-0.016}
wd=${2:-0.1}
beta2=${3:-0.8}
epsilon=${4:-1e-15}
seed=${5:-42}

ALLOW_DIRTY=1 python -u run_hydra.py \
    model=gpt-tiny \
    training=fineweb1B_tiny \
    data=fineweb1B \
    optimizer=dist-shampoo \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.beta2=${beta2}" \
    "optimizer.optimizer_params.epsilon=${epsilon}" \
    "seed=${seed}"
