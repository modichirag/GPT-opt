# GPT-opt

Small package for testing optimization methods for training GPT models from the Transformers library
## GPT2 Distillation Example

### Create a virtual environment and activate:
```bash
python3.9 -m venv viams
source viams/bin/activate
python3.9 -m pip install -e .
```

### Run Example:
```bash
python3.9 gpt_distill/gpt_distill.py --config configs/shakespeare_fi_med.yaml
```

### Or using Slurm:
```bash
./submit.sh configs/shakespeare_fi_med.yaml
```

### Plot Results:
```bash
python3.9 gpt_distill/plot_gpt.py --config configs/shakespeare_fi_med.yaml
```

### Re-run All Experiments in the Paper:
```bash
python3.9 gpt_distill/plot_gpt.py --config configs/shakespeare_fi_med.yaml
python3.9 gpt_distill/plot_gpt.py --config configs/ptb_text_med-j.yaml
python3.9 gpt_distill/plot_gpt.py --config configs/wikitext-2.yaml
```

## Poisson Regression

### Re-run Experiments:
```bash
python3.9 poisson/poisson-reg-sens.py
```
