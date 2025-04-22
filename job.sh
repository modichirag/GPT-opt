#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
#module load cuda cudnn gcc
source /mnt/home/cmodi/envs/torchlatest/bin/activate
module list 

#time torchrun --standalone --nproc_per_node=1 run.py --config configs/finewebmini.yaml
time torchrun --standalone --nproc_per_node=1 run.py --config configs/shakespeare.yaml
