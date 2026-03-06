#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --constraint=a100
#SBATCH --output=slurm_logs/slurm_job_%j.out
#SBATCH --error=slurm_logs/slurm_job_%j.err

echo "Starting sbatch.sh script on node $(hostname)..."
echo "Job ID: $SLURM_JOB_ID"
echo "TASK_FILE: $TASK_FILE"

# Ensure the directories exist
mkdir -p slurm_logs disbatch_logs

module load disBatch
module load python
source venv/bin/activate

echo "Starting disBatch with task file $TASK_FILE and log prefix 'disbatch_logs'..."
disBatch $TASK_FILE --prefix "disbatch_logs"
echo "disBatch job completed for Job ID $SLURM_JOB_ID"

# Wait for all commands to finish
wait

# Move any files starting with slurm_job_${SLURM_JOB_ID} to the specified directory
mv slurm_job_${SLURM_JOB_ID}_* slurm_logs/ 2>/dev/null || true
