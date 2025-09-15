# GPT-opt
Small package for testing optimization methods for training GPT models from the Transformers library.
This codebase is used for running the experiments in our paper, [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932).

To start, setup up the virtual enviroment and install dependencies by running
```bash
 ./setup_env.sh
```

which will create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```
Of course, you can name this virtual environment anything you want.

If using wandb, run `wandb login`.

### Run Example:
```bash
python3 run_hydra.py -cn test_hydra +training_data=shakespeare
```
In the above example, `-cn` specifies the configuration file inside `hydra_conf` (which is the configuration directory hardcoded into `run.py` itself). The `+training_data=shakespeare` adds the group `training_data/shakespeare.yaml` under the key `training_data`.

Legacy:
```bash
python3 run.py --config configs/shakespeare.yaml
```

### Plot Results:
```bash
python3 plot.py --config configs/shakespeare.yaml
```

# On the cluster

### Or using Slurm:
Make sure the virtual environment given in `./submit.sh` or `./submit_hydra.sh` is correct!
It's the same as the arguments to `run.py` itself, but without `-cn`.
```bash
./submit_hydra.sh test_hydra +training_data=shakespeare
```

Legacy version (no hydra):
```bash
./submit.sh configs/shakespeare.yaml
```

### See current jobs
```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```

# Paper Plots
Use the configuration `configs/gpt-Large-fine1B.yaml`. This uses the FineWeb dataset, and a larger GPT model.
These runs will take longer, so we recommend dividing them into separate slurm jobs. E.g., comment out all the methods except one, the `./submit.sh configs/gpt-Large-fine1B.yaml`, observe the slurm log file until training has begun (`Training with optimizer...`), then uncomment the next method and submit again. You may also wish to reduce the `#SBATCH --time` in `submit.sh`.