# GPT-opt

Small package for testing optimization methods for training GPT models from the Transformers library
## GPT2 Distillation Example

### Create a virtual environment and activate:
```bash
python3.9 -m venv gptopt
source gptopt/bin/activate
python3.9 -m pip install -e .
```

### Run Example:
```bash
python3.9 run.py --config configs/shakespeare.yaml
```

### Or using Slurm:
```bash
./submit.sh configs/shakespeare.yaml
```

### Plot Results:
```bash
python3.9 plot.py --config configs/shakespeare.yaml
```
