#!/bin/bash

CONFIG_NAME=$(basename "$1" .yaml)
EXTRA_ARGS="${@:2}"
EXTRA_SUFFIX=$(echo "$EXTRA_ARGS" | tr ' /' '-' | tr -d '.')
LOG_NAME="${CONFIG_NAME}${EXTRA_SUFFIX:+-${EXTRA_SUFFIX}}"

mkdir -p output/slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH -J ${LOG_NAME}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100-40gb
#SBATCH --exclude=workergpu027
#SBATCH -o output/slurm_logs/${LOG_NAME}.log
#SBATCH -e output/slurm_logs/${LOG_NAME}.err

module load python

# Activate environment
source venv/bin/activate

# Install the necessary packages
# python3 -m pip install -e .

export PYTHONUNBUFFERED=1

# Run the Python script with the config file
srun -u python3 -u run.py --config $1 ${EXTRA_ARGS}
EOF
