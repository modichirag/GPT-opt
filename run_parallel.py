""" Launch parallel slurm jobs to run a given list of training configs. """

import os
from typing import Dict
import subprocess
import time

import yaml


CONFIG_DIR = "configs"
LOG_DIR = "output/slurm_logs"

get_launch_script = lambda name: f"""#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH -o {LOG_DIR}/{name}.log

export OMP_NUM_THREADS=1
module load python
source gptopt/bin/activate

time torchrun --standalone --nproc_per_node=4 run.py --config {CONFIG_DIR}/{name}.yaml""" 


def get_config_path(name):
    return os.path.join(CONFIG_DIR, name + ".yaml")

def get_launch_path(name):
    return os.path.join(name + ".sh")


def run_parallel(configs: Dict[str, Dict]):

    job_ids = []
    for name, config in configs.items():

        # Write config file.
        config_path = get_config_path(name)
        config_dir = os.path.dirname(config_path)
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        with open(config_path, "w") as config_file:
            yaml.dump(config, config_file)

        # Write launch script.
        current_launch_script = get_launch_script(name)
        launch_path = get_launch_path(name)
        launch_dir = os.path.dirname(launch_path)
        if not os.path.isdir(launch_dir):
            os.makedirs(launch_dir)
        with open(launch_path, "w") as launch_file:
            launch_file.write(current_launch_script)

        # Launch a slurm job for each individual run.
        cmd = f"sbatch {launch_path}"

        p = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = p.communicate()
        job_ids.append(stdout.strip().split()[-1])

    # Wait for slurm jobs to finish.
    while len(job_ids) > 0:
        time.sleep(10)
        still_running = []
        for job_id in job_ids:
            cmd = f"squeue -h -j {job_id}"
            squeue = subprocess.run(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if job_id in squeue.stdout:
                still_running.append(job_id)
        job_ids = list(still_running)

    # Delete launch scripts.
    for name in configs.keys():
        launch_path = get_launch_path(name)
        if os.path.isfile(launch_path):
            os.remove(launch_path)
        launch_dir = os.path.dirname(launch_path)
        if len(os.listdir(launch_dir)) == 0:
            os.rmdir(launch_dir)
