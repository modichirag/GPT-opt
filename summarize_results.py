"""
Print the best validation loss of each algorithm in an experiment directory, and the configuration that
achieved it. Each subdirectory of the experiment directory holds one training run, and the subdirectory
name up to the first underscore is the algorithm name. If the subdirectory name ends with an integer
(e.g. muon_0.03_42), that integer is the seed, and losses are averaged over runs that differ only in the
seed. The validation loss of a run is the minimum over training.

Usage: python summarize_results.py gptopt/outputs/54_fineweb1B_final_comparison
"""

import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np


results_dir = sys.argv[1]

# Collect the best validation loss of each run, grouped by configuration (run name without seed).
config_losses = defaultdict(list)
run_losses = {}
for run_name in sorted(os.listdir(results_dir)):
    run_dir = os.path.join(results_dir, run_name)
    if not os.path.isdir(run_dir):
        continue
    config_name, _, suffix = run_name.rpartition("_")
    if not (config_name and suffix.isdigit()):
        config_name = run_name
    for results_path in glob.glob(os.path.join(run_dir, "*.json")):
        with open(results_path, "r") as results_file:
            val_losses = json.load(results_file)["val_losses"]
        if len(val_losses) > 0:
            config_losses[config_name].append(min(val_losses))
            run_losses[run_name] = min(val_losses)

# For each algorithm, find the configuration with the lowest mean validation loss over seeds.
best = {}
for config_name, losses in config_losses.items():
    alg = config_name.split("_")[0]
    mean = float(np.mean(losses))
    if alg not in best or mean < best[alg][0]:
        best[alg] = (mean, float(np.std(losses)), len(losses), config_name)

print("Best configuration of each algorithm:")
width = max(map(len, best), default=0)
for alg, (mean, std, n, config_name) in sorted(best.items(), key=lambda item: item[1][0]):
    print(f"{alg:{width}}  {mean:.4f} +/- {std:.4f}  (n={n})  {config_name}")

# Rank all individual runs.
print("\nAll runs:")
width = len(str(len(run_losses)))
for rank, (run_name, val_loss) in enumerate(sorted(run_losses.items(), key=lambda item: item[1]), start=1):
    print(f"{rank:{width}}.  {val_loss:.4f}  {run_name}")
