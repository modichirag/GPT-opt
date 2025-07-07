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

def plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap,  val=False):
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
    for name, data in methods.items():
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])  # Sort by learning rate
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, alpha= 0.85, label=name, color=colormap[name], linestyle=linestylemap[name], linewidth=2)
    import pdb; pdb.set_trace()
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
    # ax.set_ylim(bottom=3.0, top=4.5)
    # ax.set_xlim(0.0003, 0.05)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')

def main(config_file=None):
    default_config = get_default_config()
    if config_file:
        config = load_config(default_config, config_file)
    outfilename = config_file.replace("configs/", "").replace('.yaml', '')
    output_dir = f"gptopt/outputs/{outfilename}"
    outputs = load_outputs(output_dir)

    print(f"Loaded {len(outputs)} outputs from {output_dir}")

    for output in outputs:  # Smoothing
        smoothen_dict(output, num_points=100, beta =0.05)

# Pastel Scheme
# adam: #FFB3BA (Soft Red)
# muon-polexp: #B3CDE3 (Light Blue)
# muon-jiacheng: #C3B1E1 (Lavender)
# muon-keller: #B3E2CD (Mint Green)
# 6. Cool Tones Scheme
# adam: #61ACE5 (Sky Blue)
# muon-polexp: #00518F (Navy Blue)
# muon-jiacheng: #8A2BE2 (Purple)
# muon-keller: #008080 (Teal)
# 4. Nature-Inspired Scheme
# adam: #FFB400 (Golden Yellow - Sun)
# muon-polexp: #00518F (Ocean Blue - Water)
# muon-jiacheng: #8A2BE2 (Purple - Flowers)
# muon-keller: #228B22 (Forest Green - Trees)
    colormap = {'sgd-m': '#B3CBB9',
                'sgd-sch': '#B3CBB9',
                'adam': '#00518F',
                'adamw': '#00518F',  # Oragne'#FF6B35',
                'adam-sch': '#FF6B35',
                'momo': '#61ACE5',
                'muon-PolarExp': 'k',
                'muon-polarexpress': 'k',
                'muon-pe': 'k',  # #B3CBB9, darkish blue
                'muon-You': '#8A2BE2',  # Added a new color for "muon" (blue-violet)
                'muon*': '#228B22',
                'muon-Newtonschultz': '#008000',
                'muon-Jordan': '#FF0000',
    }
    linestylemap = {'momo': None,
                    'sgd-m': None,
                    'sgd-sch': '--',
                    'muon-pe': '--',
                    'muon-PolarExp': None,
                    'muon-polarexpress': None,
                    'adam': None,
                    'adamw': None,
                    'adam-sch': '--',
                    'muon-You': ':',
                    'muon*': None,
                    'muon-Newtonschultz': None,
                    'muon-Jordan': '-.',
    }

    # Collect learning rate ranges for each method
    lr_ranges = {}
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        if 'muon*' in name:
            name = 'muon*'
            output['name'] = 'muon*' + '-lr-' + lr  # Update the name to 'muon*'
        elif 'muon-compact' in name:
            name = 'muon-PolarExp'
            output['name'] = name + '-lr-' + lr
        elif 'muon-keller' in name:
            name = 'muon-Jordan'
            output['name'] = name + '-lr-' + lr
        elif 'muon-jiacheng' in name:
            name = 'muon-You'
            output['name'] = name + '-lr-' + lr
        lr = float(lr)
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
        name, lr = output['name'].split('-lr-')
        if 'val_losses' not in output:
            continue
        final_val_loss = output['val_losses'][-1]
        if name not in best_outputs or final_val_loss < best_outputs[name]['val_losses'][-1]:
            best_outputs[name] = output
            lr = float(lr)
            best_lr[name] = [lr, lr] 
    
    for name, output in best_outputs.items():
        print(f"Best {name}-{best_lr[name][0]} final val loss: {output['val_losses'][-1]}")
    # print(f"Best {name} lr: {lr}")
    # Plot final loss vs learning rate
    plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap)
    plot_final_loss_vs_lr(outputs, colormap, outfilename, linestylemap, val=True)
    # Plot loss
    selected_outputs = list(best_outputs.values())
    get_alpha_from_lr = lambda lr, lr_range: 0.85
    initial_loss = selected_outputs[0]['val_losses'][0] if selected_outputs and 'val_losses' in selected_outputs[0] else 1.0  # Default to 1.0 if not available
    upper_bound = initial_loss*1.0  # Set upper bound to 70% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, selected_outputs, config['training_params']['num_epochs'], 'val_losses', 'Validation Loss', colormap, linestylemap, best_lr, get_alpha_from_lr)
    lower_bound = min(min(output['val_losses']) for output in selected_outputs if 'val_losses' in output)
    ax.set_ylim(lower_bound*0.975, upper_bound) 
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
    plot_data(ax, selected_outputs, config['training_params']['num_epochs'], 'losses', 'Loss', colormap, linestylemap, best_lr, get_alpha_from_lr, time = True)
    ax.set_ylim(lower_bound*0.975, upper_bound)  # Set the upper bound
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('config', type=str, nargs='?', help='Path to config file', default=None)

    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)



