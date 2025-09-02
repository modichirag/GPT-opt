#!/bin/bash

CONFIG_NAME=$(basename "$1" .yaml)

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus-per-node=2
# SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH -C h100
#SBATCH --mem=80G
#SBATCH --nodes=1 
#SBATCH --partition=gpu
#SBATCH -o outputs/slurm_logs/${CONFIG_NAME}_%j.log

export OMP_NUM_THREADS=1

# Activate environment
source .venv/bin/activate

# Install the necessary packages
python -m pip install -e .

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=2 run.py --config $1
EOF