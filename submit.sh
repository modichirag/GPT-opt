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
# module load modules/2.4-alpha2

# Activate environment
source gptopt-env/bin/activate

# Install the necessary packages
python -m pip install -e .

# Run the Python script with the config file
python3 run.py --config $1
EOF
