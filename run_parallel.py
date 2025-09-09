""" Launch parallel slurm jobs to run a given list of training configs. """

import os
import subprocess
from typing import Dict

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
#SBATCH --time=12:00:00
#SBATCH -o {LOG_DIR}/{name}.log

export OMP_NUM_THREADS=1
module load python
source gptopt/bin/activate

time torchrun --standalone --nproc_per_node=4 run.py --config {CONFIG_DIR}/{name}.yaml""" 


def run_parallel(configs: Dict[str, Dict]):

    processes = []
    for name, config in configs.items():

        # Write config file.
        config_path = os.path.join(CONFIG_DIR, name + ".yaml")
        config_dir = os.path.dirname(config_path)
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        with open(config_path, "w") as config_file:
            yaml.dump(config, config_file)

        # Write launch script.
        current_launch_script = get_launch_script(name)
        launch_path = os.path.join(name + ".sh")
        launch_dir = os.path.dirname(launch_path)
        if not os.path.isdir(launch_dir):
            os.makedirs(launch_dir)
        with open(launch_path, "w") as launch_file:
            launch_file.write(current_launch_script)

        # Launch a slurm job for each individual run.
        cmd = f"sbatch {launch_path}".split()
        processes.append(subprocess.Popen(cmd))

    # Wait for slurm jobs to finish.
    # TODO: This is not the right way to wait for the processes to finish. This only
    # waits for the sbatch command to finish, which happens right away. Need to grab the
    # slurm job id and wait for that.
    for process in processes:
        process.wait()
