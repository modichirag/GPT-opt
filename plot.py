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
import matplotlib as mpl

# Central style configuration
def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,          # Increased from 12
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,    # Increased from 10
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 2.0,    # Increased from 1.5
        "lines.linewidth": 4.8,   # Global default line width (thicker)
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    # Disable LaTeX to avoid issues after earlier reset
    plt.rcParams["text.usetex"] = False

# Apply initially (in case functions used standalone)
apply_style()

def load_outputs(output_dir):
    outputs = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'r') as file:
                output = json.load(file)
                outputs.append(output)
    return outputs


def plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, val=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}

    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        if val:
            if 'val_losses' not in output:
                continue
            final_loss = output['val_losses'][-1]
        else:
            final_loss = output['losses'][-1]
        if name not in methods:
            methods[name] = {'lrs': [], 'losses': []}
        methods[name]['lrs'].append(lr)
        methods[name]['losses'].append(final_loss)
        if val:
            print(name, " -lr -", lr, " -loss-", final_loss)

    for output in outputs:
        name, lr = output['name'].split('-lr-')
        if 'teach_losses' in output and 'teach_losses' not in methods:
            methods['teacher'] = {'losses': []}
            methods['teacher']['losses'] = np.mean(output['teach_losses']) * np.ones(len(output['losses']))
            methods['teacher']['lrs'] = methods[name]['lrs']

    lower_bound = 100.0
    upper_bound = 0.0
    for name, data in methods.items():
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, label=name,
                color=colormap.get(name, '#000000'),
                linestyle=linestylemap.get(name, None),
                linewidth=3.0)  # Thicker explicit lines
        current_ub = np.max(sorted_losses)
        current_lb = np.min(sorted_losses)
        if current_ub > upper_bound:
            upper_bound = current_ub
        if current_lb < lower_bound:
            lower_bound = current_lb
    upper_bound *= 1.1
    upper_bound = min(upper_bound, 10.0)
    lower_bound *= 0.95
    ax.set_xscale('log')
    ax.set_ylim([lower_bound, upper_bound])
    ax.set_xlabel('Learning Rate')
    if val:
        ax.set_ylabel('Final Validation Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens-val.pdf'
    else:
        ax.set_ylabel('Final Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens.pdf'
    ax.legend(loc='upper right')
    ax.grid(axis='both', lw=0.4, ls='--', zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')


def plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, wallclock=False, val=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    tuned_methods = {}

    field = 'val_losses' if val else 'losses'
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        final_loss = float(output[field][-1])
        if name not in tuned_methods:
            tuned_methods[name] = {'best_loss': final_loss, 'best_lr': lr, 'outputs': dict(output)}
        else:
            if final_loss < tuned_methods[name]['best_loss'] or np.isnan(tuned_methods[name]['best_loss']):
                tuned_methods[name]['best_loss'] = final_loss
                tuned_methods[name]['best_lr'] = lr
                tuned_methods[name]['outputs'] = dict(output)
    print("Best Validation losses:" if val else "Best losses:")
    for name in tuned_methods:
        print(f"{name}: {tuned_methods[name]['best_loss']} at lr {tuned_methods[name]['best_lr']}")

    tuned_outputs = [tuned_methods[name]['outputs'] for name in tuned_methods]
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, field, 'Loss', colormap, linestylemap,
              lr_ranges, get_alpha_from_lr, wallclock=wallclock)
    upper_bound = np.max([output[field][round(0.2 * len(output[field]))] for output in tuned_outputs])
    lower_bound = 100
    for output in tuned_outputs:
        lower_bound = float(np.minimum(lower_bound, np.min(output[field])))
    upper_bound = min(upper_bound, 10.0) if not np.isnan(upper_bound) else 10.0
    lower_bound = max(lower_bound, 3.0) if not np.isnan(lower_bound) else 3.0
    lower_bound *= 0.95
    ax.legend(loc='upper right')
    ax.set_ylim(lower_bound, upper_bound)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    suffix = "_tuned"
    if wallclock:
        suffix += "_wallclock"
    if val:
        suffix += "_val"
    fig.savefig("figures/" + outfilename + suffix + '.pdf', format='pdf', bbox_inches='tight')


def main(config_file=None):
    default_config = get_default_config()
    if config_file:
        config = load_config(default_config, config_file)
    outfilename = config_file.replace("configs/", "").replace('.yaml', '')
    output_dir = f"gptopt/outputs/{outfilename}"
    outputs = load_outputs(output_dir)

    print(f"Loaded {len(outputs)} outputs from {output_dir}")

    for output in outputs:
        smoothen_dict(output, num_points=None, beta=0.05)

    colormap = {
        'sgd-m': '#B3CBB9',
        'adamw': '#FF6B35',
        'iams': '#61ACE5',
        'iams-adam': '#1B75BC',
        'teacher': 'k',
        'sgd-schedulep': '#FF00FF',
        'adamw-schedulep': '#8B008B',
        'sgd-schedulefree': '#008000',
        'adamw-schedulefree': '#006400',
    }

    linestylemap = {
        'iams': None,
        'sgd-m': None,
        'sgd-sch': '--',
        'teacher': '--',
        'iams-adam': None,
        'adam': None,
        'adamw': '--',
        'adam-sch': '--',
        'muon': None,
        'muon-nonlmo': None,
        'sgd-schedulep': None,
        'sgd-schedulefree': None,
        'muon-l2_prod': None,
        'adamw-schedulefree': '--',
        'adamw-schedulep': '--',
        'muon-nonlmo-rms': None,
        'muon-l2_prod-rms': None,
        'muon-nonlmo-l2_prod-rms': None,
        'sign-gd': None,
    }

    lr_ranges = {}
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        if name not in lr_ranges:
            lr_ranges[name] = [lr, lr]
        else:
            lr_ranges[name][0] = min(lr_ranges[name][0], lr)
            lr_ranges[name][1] = max(lr_ranges[name][1], lr)

    # Reset then re-apply thicker style
    mpl.rcParams.update(mpl.rcParamsDefault)
    apply_style()

    plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename)
    plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, val=True)

    initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0
    upper_bound = initial_loss * 1.2
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plot_data(ax, outputs,  config['training_params']['num_epochs'], 'losses', 'Loss',
              colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    lower_bound *= 0.95
    ax.set_ylim(lower_bound, upper_bound)
    ax.legend(loc='upper right')
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    for method_subset in [['sgd-m', 'sgd-sch', 'iams', 'sgd-schedulep'],
                          ['adam', 'adam-sch', 'iams-adam', 'adamw-schedulep']]:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        subset_outputs = [output for output in outputs if output['name'].split('-lr-')[0] in method_subset]
        plot_data(ax, subset_outputs, config['training_params']['num_epochs'], 'learning_rates',
                  'Learning rate', colormap, linestylemap, lr_ranges, get_alpha_from_lr)
        ax.legend(loc='upper right')
        fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
        name = '-lr' if 'sgd-m' in method_subset else '-lr-adam'
        fig.savefig('figures/' + outfilename + name + '.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plotted_methods = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(handle) for handle in handles]
    for handle in legend_handles:
        handle.set_alpha(1.0)
    ax.legend(legend_handles, labels, loc='upper right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '-step_size-.pdf', format='pdf', bbox_inches='tight')

    plot_tuned_curves(outputs, colormap, linestylemap, outfilename,
                      config['training_params']['num_epochs'], wallclock=False, val=False)
    plot_tuned_curves(outputs, colormap, linestylemap, outfilename,
                      config['training_params']['num_epochs'], wallclock=False, val=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('config', type=str, nargs='?', help='Path to config file', default=None)
    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)
