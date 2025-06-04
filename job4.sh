#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH -o output/slurm_logs/fineweb1B_constant-linear.log

export OMP_NUM_THREADS=1

#module load modules/2.3-20240529
#module load cuda cudnn gcc
#source /mnt/home/cmodi/envs/torchlatest/bin/activate
#module list 
module load python
source gptopt/bin/activate

time torchrun --standalone --nproc_per_node=4 run.py --config configs/fineweb1B.yaml
