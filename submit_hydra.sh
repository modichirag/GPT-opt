#!/bin/bash

CONFIG_NAME=$1

mkdir -p outputs/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${CONFIG_NAME}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --constraint=h100
#SBATCH --mem=80G
#SBATCH --nodes=1 
#SBATCH --partition=gpu
#SBATCH --exclude=workergpu027
#SBATCH -o outputs/slurm_logs/${CONFIG_NAME}_%j.log

module load python
export OMP_NUM_THREADS=1

# Activate environment
source .venv/bin/activate

# Install the necessary packages
python3 -m pip install -e .

export PYTHONUNBUFFERED=1

# Run the Python script with the config file
#### time torchrun --standalone --nproc_per_node=2 run_hydra.py -cn $@
srun -u python3 -u run_hydra.py -cn $@
EOF
