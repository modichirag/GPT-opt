#!/bin/bash

CONFIG_NAME=$(basename "$1" .yaml)

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=150:00:00
#SBATCH --partition=gpu
#SBATCH -o output/slurm_logs/${CONFIG_NAME}.log

module load python

# Activate environment
source gptopt/bin/activate

# Install the necessary packages
python3 -m pip install -e .

# Run the Python script with the config file
time torchrun --standalone --nproc_per_node=2 run.py --config $1
EOF
