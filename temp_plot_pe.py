import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.utils import get_default_config, load_config
from gptopt.plot_utils import get_alpha_from_lr, percentage_of_epoch, plot_data, plot_step_size_and_lr, smoothen_dict
import copy
import json
import os
import numpy as np
import glob


algs = ["muon", "muon-nonlmo", "muon-l2_prod", "muon-nonlmo-l2_prod", "muon-nonlmo-nuc_past"]
lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
families = ["ns", "pe"]
subdirs = ["fineweb1B_constant-linear", "fineweb1B_polar_express"]
linestyles = ["-", "--"]
results_dir = "gptopt/outputs"
outfile = "polar_express_comparison.pdf"

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=10)

colormap = {'sgd-m': '#B3CBB9',
            'sgd-sch': '#B3CBB9',
            'adam': '#FF6B35',
            'adamw': '#FF6B35',
            'adam-sch': '#FF6B35',
            'momo': '#61ACE5',
            'momo-adam': '#00518F',
            'teacher': 'k',
            'muon': '#8A2BE2',  # Added a new color for "muon" (blue-violet)
            'muon-nonlmo': '#000000',
            'muon-nonlmo-nuc_past': '#808080',
            'muon-l2_prod': '#008000',
            'muon-nonlmo-l2_prod': '#FF0000',
            'muon-rms': '#7FFFD4',
            'muon-nonlmo-rms': '#BE6400',
            'muon-l2_prod-rms': '#FF00FF',
            'muon-nonlmo-l2_prod-rms': '#FFD700',
}


def load_results():
    """Load all individual output files from a directory."""
    results = {}
    for family, subdir in zip(families, subdirs):
        results[family] = {}
        for alg in algs:
            results[family][alg] = {}
            for lr in lrs:
                file_prefix = f"{alg}-lr-{lr}-*.json"
                filenames = glob.glob(os.path.join(results_dir, subdir, file_prefix))
                filename = filenames[0]
                with open(filename, "r") as f:
                    results[family][alg][lr] = json.load(f)

    return results


def main():
    results = load_results()

    # Michael: Temparily resetting matplotlib settings to default so that latex doesn't
    # need to be used for plot formatting. Was giving me an error.
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Get final losses.
    final_losses = {}
    for family in families:
        final_losses[family] = {}
        for alg in algs:
            final_losses[family][alg] = {}
            for lr in lrs:
                final_losses[family][alg][lr] = results[family][alg][lr]["losses"][-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each method as a line
    for i, family in enumerate(families):
        linestyle = linestyles[i]
        for alg in algs:
            losses = [results[family][alg][lr]["losses"][-1] for lr in lrs]
            plot_kwargs = {"label": f"{alg}"} if i == 0 else {}
            ax.plot(lrs, losses, color=colormap[alg], linestyle=linestyle, linewidth=2, **plot_kwargs)

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Loss')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    ax.legend(loc='upper right', fontsize=10)

    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(outfile, format='pdf', bbox_inches='tight')

    # Print gaps.
    for alg in algs:
        gaps = [results[families[0]][alg][lr]["losses"][-1] - results[families[1]][alg][lr]["losses"][-1] for lr in lrs]
        avg_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        print(f"Max, avg gap for {alg}: {max_gap}, {avg_gap}")


if __name__ == "__main__":
    main()
