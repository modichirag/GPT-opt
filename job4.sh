#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
#module load cuda cudnn gcc
source /mnt/home/cmodi/envs/torchlatest/bin/activate
module list 

#time torchrun --standalone --nproc_per_node=4 run_distributed.py --config configs/finewebmini.yaml 
time torchrun --standalone --nproc_per_node=4 run_distributed.py --config configs/fineweb1B.yaml --suffix nognorm
#time torchrun --standalone --nproc_per_node=4 run_distributed.py --config configs/fineweb10B.yaml
#time torchrun --standalone --nproc_per_node=4 run_distributed.py --config configs/shakespeare-local.yaml
