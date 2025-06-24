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

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=10)

def load_outputs(output_dir):
    """Load all individual output files from a directory."""
    outputs = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'r') as file:
                output = json.load(file)
                outputs.append(output)
    return outputs


def plot_final_loss_vs_lr(outputs, colormap, outfilename, val=False):
    """Plot final loss versus learning rate as lines for each method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}

    # Group final losses and learning rates by method
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
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
        ax.plot(sorted_lrs, sorted_losses, label=name, color=colormap[name], linewidth=2)
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
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')


def plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, wallclock=False, val=False):
    """Plot loss curves of tuned methods."""
    fig, ax = plt.subplots(figsize=(6, 4))
    tuned_methods = {}

    # Find best lr for each method.
    field = 'val_losses' if val else 'losses'
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        final_loss = float(output[field][-1])
        if name not in tuned_methods:
            tuned_methods[name] = {'best_loss': final_loss, 'best_lr': lr, 'outputs': dict(output)}
        else:
            if final_loss < tuned_methods[name]['best_loss']:
                tuned_methods[name]['best_loss'] = final_loss
                tuned_methods[name]['best_lr'] = lr
                tuned_methods[name]['outputs'] = dict(output)

    # Plot loss of tuned methods.
    tuned_outputs = [tuned_methods[name]['outputs'] for name in tuned_methods]
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, field, 'Loss', colormap, linestylemap, lr_ranges, get_alpha_from_lr, wallclock=wallclock)
    upper_bound = np.max([output[field][round(0.2 * len(output[field]))] for output in tuned_outputs])
    upper_bound = min(upper_bound, 10.0) if not np.isnan(upper_bound) else 10.0
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(3.2, upper_bound)
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

    for output in outputs:  # Smoothing
        smoothen_dict(output, num_points=None, beta =0.05)

    colormap = {'sgd-m': '#B3CBB9',
                'sgd-sch': '#B3CBB9',
                'adam': '#FF6B35',
                'adamw': '#FF6B35',
                'adam-sch': '#FF6B35',
                'momo': '#61ACE5',
                'momo-adam': '#00518F',
                'teacher': 'k',
                'muon': '#8A2BE2',  # Added a new color for "muon" (blue-violet)
                'muon-nonlmo': '#FFFF00',
                'muon-nonlmo-fro_approx': '#000000',
                'muon-nonlmo-nuc_fro': '#000000',
                'muon-nonlmo-nuc_past': '#808080',
                'muon-l2_prod': '#008000',
                'muon-nonlmo-l2_prod': '#FF0000',
                'muon-rms': '#7FFFD4',
                'muon-nonlmo-rms': '#BE6400',
                'muon-l2_prod-rms': '#FF00FF',
                'muon-nonlmo-l2_prod-rms': '#FFD700',
                'sign-gd': '#61ACE5',
                'sign-gd-nonlmo': '#00518F',
    }
    linestylemap = {'momo': None,
                    'sgd-m': None,
                    'sgd-sch': '--',
                    'teacher': '--',
                    'momo-adam': None,
                    'adam': None,
                    'adamw': None,
                    'adam-sch': '--',
                    'muon': None,
                    'muon-nonlmo': None,
                    'muon-nonlmo-fro_approx': None,
                    'muon-nonlmo-nuc_fro': None,
                    'muon-nonlmo-nuc_past': None,
                    'muon-l2_prod': None,
                    'muon-nonlmo-l2_prod': None,
                    'muon-rms': None,
                    'muon-nonlmo-rms': None,
                    'muon-l2_prod-rms': None,
                    'muon-nonlmo-l2_prod-rms': None,
                    'sign-gd': None,
                    'sign-gd-nonlmo': None,
    }

    # Collect learning rate ranges for each method
    lr_ranges = {}
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
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
    plot_final_loss_vs_lr(outputs, colormap, outfilename)
    plot_final_loss_vs_lr(outputs, colormap, outfilename, val=True)

    # Plot loss
    initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0  # Default to 1.0 if not available
    upper_bound = initial_loss * 1.2  # Set upper bound to 20% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, outputs,  config['training_params']['num_epochs'], 'losses', 'Loss', colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    ax.set_ylim(lower_bound, upper_bound) # Set the upper bound
    ax.legend(loc='upper right', fontsize=10)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')


    # Plot learning rates
    for method_subset in [['sgd-m', 'sgd-sch', 'momo'], ['adam', 'adam-sch', 'momo-adam']]:
        fig, ax = plt.subplots(figsize=(4, 3))
        subset_outputs = [output for output in outputs if output['name'].split('-lr-')[0] in method_subset]
        plot_data(ax, subset_outputs, config['training_params']['num_epochs'], 'learning_rates', 'Learning rate', colormap, linestylemap, lr_ranges,  get_alpha_from_lr)
        ax.legend(loc='upper right', fontsize=10)
        fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
        name = 'figures/lr-' if 'sgd-m' in method_subset else 'figures/lr-adam-'
        fig.savefig(name + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # Plot step size lists
    fig, ax = plt.subplots(figsize=(4, 3))
    plotted_methods = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
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
    plot_tuned_curves(outputs, colormap, linestylemap, outfilename, config['training_params']['num_epochs'], wallclock=False, val=False)
    plot_tuned_curves(outputs, colormap, linestylemap, outfilename, config['training_params']['num_epochs'], wallclock=False, val=True)
    #plot_tuned_curves(outputs, colormap, linestylemap, outfilename, config['training_params']['num_epochs'], wallclock=True, val=False)
    #plot_tuned_curves(outputs, colormap, linestylemap, outfilename, config['training_params']['num_epochs'], wallclock=True, val=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)

    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)



