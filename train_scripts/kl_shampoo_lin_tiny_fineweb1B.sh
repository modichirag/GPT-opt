#!/bin/bash
lr=${1:-0.003}
wd=${2:-0.1}
shampoo_beta=${3:-0.95}
precondition_frequency=${4:-10}
seed=${5:-42}

ALLOW_DIRTY=1 python -u run_hydra.py \
    model=gpt-tiny \
    training=fineweb1B_tiny \
    data=fineweb1B \
    optimizer=kl-shampoo \
    logging=default \
    "optimizer.optimizer_params.lr=${lr}" \
    "optimizer.optimizer_params.weight_decay=${wd}" \
    "optimizer.optimizer_params.shampoo_beta=${shampoo_beta}" \
    "optimizer.optimizer_params.precondition_frequency=${precondition_frequency}" \
    "seed=${seed}"
