#!/bin/bash

CONFIG_NAME=$1

mkdir -p output/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --exclude=workergpu027
#SBATCH -o output/slurm_logs/${CONFIG_NAME}_%j.log
#SBATCH -e output/slurm_logs/${CONFIG_NAME}_%j.err

module load python

# Activate environment
source venv/bin/activate

# Install the necessary packages
# python3 -m pip install -e .

export PYTHONUNBUFFERED=1

# Run the Python script with the config file
srun -u python3 -u run_hydra.py -cn $@
EOF
