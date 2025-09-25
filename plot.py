from functools import reduce
import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.utils import get_default_config, load_config
from gptopt.plot_utils import get_alpha_from_lr, plot_data, plot_step_size_and_lr, smoothen_dict, get_lr_and_name
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

def load_output_folder(experiment_results_folder):
    outputs = []
    for root, _, files in os.walk(experiment_results_folder):
        for file_name in files:
            if file_name.startswith("logs") and file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    logs = json.load(file)
                with open(os.path.join(root, ".hydra/config.yaml"), 'r') as file:
                    config = yaml.safe_load(file)
                outputs.append(dict(config=config, logs=logs))
    return outputs

def plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap, val=False, y_top_lim=None):
    """Plot final loss versus learning rate as lines for each method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = {}

    # Group final losses and learning rates by method
    for output in outputs:
        name, lr = get_lr_and_name(output)
        lr = float(lr)
        if val:
            if 'val_losses' not in output['logs']:
                continue
            final_loss = output['logs']['val_losses'][-1]
        else:
            final_loss = output['logs']['losses'][-1]  # Get the final loss
        if name not in methods:
            methods[name] = {'lrs': [], 'losses': []}
        methods[name]['lrs'].append(lr)
        methods[name]['losses'].append(final_loss)

    # Plot each method as a line
    for name, data in methods.items():
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])  # Sort by learning rate
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        if len(set(sorted_lrs)) < len(sorted_lrs):
            print(f"Warning: Duplicate learning rates found for method {name}. This may affect the line plot.")
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, alpha= 0.85, label=name, color=colormap.get(name, None), linestyle=linestylemap.get(name, None), linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    if val:
        ax.set_ylabel('Final Validation Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens'  + '-val' + '.pdf'
    else:
        ax.set_ylabel('Final Loss')
        plotfile = 'figures/' + outfilename + '-lr-sens' + '.pdf'
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
    if y_top_lim is not None:
        ax.set_ylim(bottom=3.35, top=y_top_lim)
    # ax.set_ylim(bottom=3.0, top=4.5)
    # ax.set_xlim(0.0003, 0.05)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')

def main(outputs, outfilename, y_top_lim_lrs=None, y_top_vs_time=None):
    for output in outputs:  # Smoothing
        smoothen_dict(output['logs'], num_points=100, beta =0.05)


    colormap = {'sgd-m': '#B3CBB9',
                'sgd-sch': '#B3CBB9',
                'adam': '#00518F',
                'adamw': '#00518F',  # Oragne'#FF6B35',
                'adam-sch': '#FF6B35',
                'momo': '#61ACE5',
                'muon-polarexpress': 'k',
                'muon-You': '#8A2BE2',  # Added a new color for "muon" (blue-violet)
                'muon-Jordan': '#FF0000',
    }
    linestylemap = {'momo': None,
                    'sgd-m': None,
                    'sgd-sch': '--',
                    'muon-polarexpress': None,
                    'adam': None,
                    'adamw': None,
                    'adam-sch': '--',
                    'muon-You': ':',
                    'muon-Jordan': '-.',
    }

    # Collect learning rate ranges for each method
    lr_ranges = {}
    for output in outputs:
        name, lr = get_lr_and_name(output)
        if name not in lr_ranges:
            lr_ranges[name] = [lr, lr]
        else:
            lr_ranges[name][0] = min(lr_ranges[name][0], lr)
            lr_ranges[name][1] = max(lr_ranges[name][1], lr)         

    # Michael: Temparily resetting matplotlib settings to default so that latex doesn't
    # need to be used for plot formatting. Was giving me an error.
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)

    best_outputs = {}
    best_lr = {}
    for output in outputs:
        name, lr = get_lr_and_name(output)
        if 'val_losses' not in output['logs']:
            continue
        final_val_loss = output['logs']['val_losses'][-1]
        if name not in best_outputs or final_val_loss < best_outputs[name]['logs']['val_losses'][-1]:
            best_outputs[name] = output
            lr = float(lr)
            best_lr[name] = [lr, lr] 
    os.makedirs("figures", exist_ok=True)
    for name, output in best_outputs.items():
        print(f"Best {name}-{best_lr[name][0]} final val loss: {output['logs']['val_losses'][-1]}")
    # print(f"Best {name} lr: {lr}")
    # Plot final loss vs learning rate
    plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap, y_top_lim=y_top_lim_lrs)
    plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap, val=True, y_top_lim=y_top_lim_lrs)
    # Plot loss
    selected_outputs = list(best_outputs.values())
    get_alpha_from_lr = lambda lr, lr_range: 0.85
    initial_loss = selected_outputs[0]['logs']['val_losses'][0] if selected_outputs and 'val_losses' in selected_outputs[0]['logs'] else 1.0  # Default to 1.0 if not available
    upper_bound = initial_loss*1.0  # Set upper bound to 70% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, selected_outputs, max(o['config']['training_data']['training_params']['num_epochs'] for o in selected_outputs), 'val_losses', 'Validation Loss', colormap, linestylemap, best_lr, get_alpha_from_lr)
    lower_bound = min(min(output['logs']['val_losses']) for output in selected_outputs if 'val_losses' in output['logs'])
    ax.set_ylim(lower_bound*0.975, upper_bound) 
    if y_top_vs_time is not None:
        ax.set_ylim(top=y_top_vs_time)
    ax.tick_params(axis='both', which='major', labelsize=8)  # Set tick label font size
    ax.set_xlabel('Epoch', fontsize=10)  # Set x-axis label font size
    ax.set_ylabel('Validation Loss', fontsize=10) 
    # Set the upper bound
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=10) 
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # Legend placed next to the figure
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.8)  # Adjust right to make space for legend
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, selected_outputs, max(o['config']['training_data']['training_params']['num_epochs'] for o in selected_outputs), 'losses', 'Loss', colormap, linestylemap, best_lr, get_alpha_from_lr, time = True)
    ax.set_ylim(lower_bound*0.975, upper_bound)  # Set the upper bound
    if y_top_vs_time is not None:
        ax.set_ylim(top=y_top_vs_time)
    ax.tick_params(axis='both', which='major', labelsize=8)  # Set tick label font size
    ax.set_xlabel('Time (s)', fontsize=10)  # Set x-axis label font size
    ax.set_ylabel('Validation Loss', fontsize=10) 
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=10) 
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # Legend placed next to the figure
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.8)  # Adjust right to make space for legend
    fig.savefig('figures/' + outfilename + '-time.pdf', format='pdf', bbox_inches='tight')
    # initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0  # Default to 1.0 if not available
    # upper_bound = initial_loss * 1.2  # Set upper bound to 20% above the initial loss
    # fig, ax = plt.subplots(figsize=(4, 3))
    # plot_data(ax, outputs,  config['training_params']['num_epochs'], 'losses', 'Loss', colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    # lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    # ax.set_ylim(lower_bound, upper_bound) # Set the upper bound
    # ax.legend(loc='upper right', fontsize=10)
    # fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    # fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')


    # # Plot learning rates
    # for method_subset in [['sgd-m', 'sgd-sch', 'momo'], ['adam', 'adam-sch', 'momo-adam']]:
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     subset_outputs = [output for output in outputs if output['name'].split('-lr-')[0] in method_subset]
    #     plot_data(ax, subset_outputs, config['training_params']['num_epochs'], 'learning_rates', 'Learning rate', colormap, linestylemap, lr_ranges,  get_alpha_from_lr)
    #     ax.legend(loc='upper right', fontsize=10)
    #     fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
    #     name = 'figures/lr-' if 'sgd-m' in method_subset else 'figures/lr-adam-'
    #     fig.savefig(name + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    # # Plot step size lists
    # fig, ax = plt.subplots(figsize=(4, 3))
    # plotted_methods = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    # handles, labels = ax.get_legend_handles_labels()
    # legend_handles = [copy.copy(handle) for handle in handles]
    # for handle in legend_handles:
    #     handle.set_alpha(1.0)
    # ax.legend(legend_handles, labels, loc='upper right', fontsize=10)
    # ax.set_xlabel('Step')
    # ax.set_ylabel('Learning Rate')
    # fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    # fig.savefig('figures/step_size-' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    # parser.add_argument('--results_folder', type=str, nargs='?', help='Path to results folder', default=None)

    results_folder = "outputs/hydra-results/main_run"
    outputs = load_output_folder("outputs/hydra-results/main_run")

    for weight_decay in set(output['config']['optimizer_params']['args']['weight_decay'] for output in outputs):
        small_outputs = [
            output for output in outputs 
            if output['config']['optimizer_params']['args']['weight_decay'] == weight_decay
        ]
        for nlayer in set(output['config']['gpt_model']['n_layer'] for output in small_outputs):
            smaller_outputs = [
                output for output in small_outputs 
                if output['config']['gpt_model']['n_layer'] == nlayer
            ]
            assert len(small_outputs) > 0
            outfilename = os.path.basename(results_folder.rstrip('/')) + f"-nl-{nlayer}" + "-wd-" + str(weight_decay)
            print(f"Loaded {len(smaller_outputs)} outputs from {results_folder}")
            main(smaller_outputs, outfilename, y_top_lim_lrs=3.7, y_top_vs_time=4.5)
