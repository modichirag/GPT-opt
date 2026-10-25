import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.plot_utils import get_alpha_from_lr, percentage_of_epoch, plot_data, plot_step_size_and_lr, smoothen_dict
import copy
import glob
import json
import math
import os
import numpy as np
from collections import defaultdict

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=10)

def load_outputs(results_dir, config_dir):
    """
    Load the output of each run in results_dir, where each subdirectory holds one run. The method and (base)
    learning rate of each run are read from the run's config, config_dir/<run_name>.yaml. Runs of the same
    optimizer that differ in other optimizer settings (e.g. angd_lr_scale) are treated as different methods,
    labeled with the settings that vary, so that each gets its own curve.
    """
    outputs = []
    for run_name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        results_paths = glob.glob(os.path.join(run_dir, "*.json"))
        if len(results_paths) == 0:
            print(f"Skipping {run_name}, no output file.")
            continue
        assert len(results_paths) == 1
        with open(results_paths[0], 'r') as file:
            output = json.load(file)
        with open(os.path.join(config_dir, run_name + ".yaml"), 'r') as file:
            config = yaml.safe_load(file)

        assert len(config['optimizer_params']) == 1
        assert len(config['optimizer_params'][0]['lr']) == 1
        output['lr'] = float(config['optimizer_params'][0]['lr'][0])
        output['run_name'] = run_name
        output['config'] = config
        outputs.append(output)

    # Find the optimizer settings other than lr that vary across runs of each optimizer.
    opt_configs = defaultdict(list)
    for output in outputs:
        opt_config = output['config']['optimizer_params'][0]
        opt_configs[opt_config['name']].append(opt_config)
    varying_keys = {}
    for name, configs in opt_configs.items():
        keys = sorted(set().union(*configs) - {'name', 'lr'})
        varying_keys[name] = [k for k in keys if len({json.dumps(c.get(k)) for c in configs}) > 1]

    # Label each run by its optimizer and varying settings.
    for output in outputs:
        opt_config = output['config']['optimizer_params'][0]
        name = opt_config['name']
        settings = [opt_config.get(k) for k in varying_keys[name]]
        output['method'] = name
        if len(settings) > 0:
            output['method'] += " (" + ", ".join(f"{k}={v}" for k, v in zip(varying_keys[name], settings)) + ")"
        # Settings can mix types (e.g. muon_lr_scale of 10.0 or "moonlight"), so sort numbers before strings.
        settings_key = [(0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v)) for v in settings]
        output['sort_key'] = (name, settings_key, output['lr'])

    return sorted(outputs, key=lambda output: output['sort_key'])


def plot_final_loss_vs_lr(outputs, outfilename, val=False):
    """Plot final loss versus learning rate as lines for each method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}

    # Group final losses and learning rates by method
    for output in outputs:
        name, lr = output['method'], output['lr']
        if val:
            if 'val_losses' not in output:
                continue
            final_loss = output['val_losses'][-1]
        else:
            final_loss = output['losses'][-1]  # Get the final loss
        if name not in methods:
            methods[name] = {'lrs': [], 'losses': []}
        methods[name]['lrs'].append(lr)
        methods[name]['losses'].append(final_loss)

    # Plot each method as a line
    lower_bound = 3.2
    upper_bound = 0.0
    for name, data in methods.items():
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])  # Sort by learning rate
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, label=name, linewidth=2)
        current_ub = np.max(sorted_losses)
        if current_ub > upper_bound:
            upper_bound = current_ub
    upper_bound *= 1.1
    upper_bound = min(upper_bound, 10.0)

    ax.set_xscale('log')
    ax.set_ylim([lower_bound, upper_bound])
    ax.set_xlabel('Learning Rate')
    if val:
        ax.set_ylabel('Final Validation Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens'  + '-val' + '.pdf'
    else:
        ax.set_ylabel('Final Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens' + '.pdf'
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')


def plot_tuned_curves(outputs, outfilename, num_epochs, wallclock=False, val=False):
    """Plot loss curves of tuned methods."""
    fig, ax = plt.subplots(figsize=(6, 4))
    tuned_methods = {}

    # Find best lr for each method.
    field = 'val_losses' if val else 'losses'
    for output in outputs:
        name, lr = output['method'], output['lr']
        final_loss = float(output[field][-1])
        if math.isnan(final_loss):
            final_loss = float('inf')
        if name not in tuned_methods:
            tuned_methods[name] = {'best_loss': final_loss, 'best_lr': lr, 'outputs': dict(output)}
        else:
            if final_loss < tuned_methods[name]['best_loss']:
                tuned_methods[name]['best_loss'] = final_loss
                tuned_methods[name]['best_lr'] = lr
                tuned_methods[name]['outputs'] = dict(output)

    for name in tuned_methods:
        print(f"Tuned {name} ({field}): {tuned_methods[name]['outputs']['run_name']}")

    # Plot loss of tuned methods.
    tuned_outputs = [tuned_methods[name]['outputs'] for name in tuned_methods]
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, field, 'Loss', lr_ranges, get_alpha_from_lr, wallclock=wallclock)
    upper_bound = np.max([output[field][round(0.2 * len(output[field]))] for output in tuned_outputs])
    upper_bound = min(upper_bound, 10.0) if not np.isnan(upper_bound) else 10.0
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(3.2, upper_bound)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    suffix = "_tuned"
    if wallclock:
        suffix += "_wallclock"
    if val:
        suffix += "_val"
    fig.savefig("figures/" + outfilename + suffix + '.pdf', format='pdf', bbox_inches='tight')


def main(results_dir):
    results_dir = results_dir.rstrip('/')
    outfilename = os.path.basename(results_dir)
    config_dir = os.path.join("configs", outfilename)
    outputs = load_outputs(results_dir, config_dir)
    print(f"Loaded {len(outputs)} outputs from {results_dir}")
    num_epochs = outputs[0]['config']['training_params']['num_epochs']
    os.makedirs('figures', exist_ok=True)

    for output in outputs:  # Smoothing
        smoothen_dict(output, num_points=None, beta =0.05)

    # Collect learning rate ranges for each method
    lr_ranges = {}
    for output in outputs:
        name, lr = output['method'], output['lr']
        if name not in lr_ranges:
            lr_ranges[name] = [lr, lr]
        else:
            lr_ranges[name][0] = min(lr_ranges[name][0], lr)
            lr_ranges[name][1] = max(lr_ranges[name][1], lr)

    # Michael: Temparily resetting matplotlib settings to default so that latex doesn't
    # need to be used for plot formatting. Was giving me an error.
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Plot final loss vs learning rate
    plot_final_loss_vs_lr(outputs, outfilename)
    plot_final_loss_vs_lr(outputs, outfilename, val=True)

    # Plot loss
    initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0  # Default to 1.0 if not available
    upper_bound = initial_loss * 1.2  # Set upper bound to 20% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, outputs,  num_epochs, 'losses', 'Loss', lr_ranges, get_alpha_from_lr)
    lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    ax.set_ylim(lower_bound, upper_bound) # Set the upper bound
    ax.legend(loc='best', fontsize=10)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')


    # Plot learning rates
    for method_subset in [['sgd-m', 'sgd-sch', 'momo'], ['adam', 'adam-sch', 'momo-adam']]:
        fig, ax = plt.subplots(figsize=(4, 3))
        subset_outputs = [output for output in outputs if output['method'] in method_subset]
        if len(subset_outputs) == 0:
            continue
        plot_data(ax, subset_outputs, num_epochs, 'learning_rates', 'Learning rate', lr_ranges,  get_alpha_from_lr)
        ax.legend(loc='best', fontsize=10)
        fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
        name = 'figures/lr-' if 'sgd-m' in method_subset else 'figures/lr-adam-'
        fig.savefig(name + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # Plot step size lists
    if any('step_size_list' in output for output in outputs):
        fig, ax = plt.subplots(figsize=(4, 3))
        plotted_methods = plot_step_size_and_lr(ax, outputs, lr_ranges, get_alpha_from_lr)
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = [copy.copy(handle) for handle in handles]
        for handle in legend_handles:
            handle.set_alpha(1.0)
        ax.legend(legend_handles, labels, loc='upper right', fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
        fig.savefig('figures/step_size-' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # Plot loss curves of tuned algorithms.
    plot_tuned_curves(outputs, outfilename, num_epochs, wallclock=False, val=False)
    plot_tuned_curves(outputs, outfilename, num_epochs, wallclock=False, val=True)
    #plot_tuned_curves(outputs, outfilename, num_epochs, wallclock=True, val=False)
    #plot_tuned_curves(outputs, outfilename, num_epochs, wallclock=True, val=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('results_dir', type=str,
                        help='Directory with one subdirectory per run, e.g. gptopt/outputs/80_angd_first_comparison. '
                             'Run configs are read from configs/<basename of results_dir>/.')

    args = parser.parse_args()
    main(args.results_dir)
