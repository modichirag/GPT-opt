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
#SBATCH -o output/slurm_logs/${CONFIG_NAME}.log

export OMP_NUM_THREADS=1
# module load modules/2.4-alpha2

# Activate environment
source gptopt-env/bin/activate

# Install the necessary packages
python -m pip install -e .

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=2 run.py --config $1
EOF
