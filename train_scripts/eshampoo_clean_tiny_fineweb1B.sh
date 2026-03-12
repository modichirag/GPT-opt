#!/bin/bash
lr=${1:-0.01}
wd=${2:-0.1}
beta2=${3:-0.99}
epsilon=${4:-1e-15}
seed=${5:-42}

ALLOW_DIRTY=1 python -u run_hydra.py \
    model=gpt-tiny \
    training=fineweb1B_tiny \
    data=fineweb1B \
    optimizer=shampoo-clean \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.beta2=${beta2}" \
    "optimizer.optimizer_params.epsilon=${epsilon}" \
    "optimizer.optimizer_params.eshampoo=true" \
    "optimizer.optimizer_params.nesterov=false" \
    "seed=${seed}"
