#!/bin/bash

CONFIG_NAME=$(basename "$1" .yaml)

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH -o output/slurm_logs/${CONFIG_NAME}.log

export OMP_NUM_THREADS=1
module load modules/2.3-20240529
module load python

# Activate environment
source gptopt/bin/activate

# Install the necessary packages
python3.9 -m pip install -e .
time torchrun --standalone --nproc_per_node=1 run.py --config $1
EOF
