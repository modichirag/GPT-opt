import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.utils import get_default_config, load_config
from gptopt.plot_utils import get_alpha_from_lr, percentage_of_epoch, plot_data, plot_step_size_and_lr, smoothen_dict
import copy
import json
import os

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
    for name, data in methods.items():
        sorted_indices = sorted(range(len(data['lrs'])), key=lambda i: data['lrs'][i])  # Sort by learning rate
        sorted_lrs = [data['lrs'][i] for i in sorted_indices]
        sorted_losses = [data['losses'][i] for i in sorted_indices]
        ax.plot(sorted_lrs, sorted_losses, label=name, color=colormap[name], linewidth=2)

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
    # ax.set_ylim(top=7)
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
        smoothen_dict(output, num_points=100, beta=0.05)

    # Define a list of colors to rotate through
    color_list = ['#B3CBB9', '#FF6B35', '#61ACE5', '#8A2BE2', '#008000', '#FF0000', '#00518F', '#FFD700']
    line_style_list = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    color_index = 0

    # Dynamically assign colors to methods starting with "muon*-<string>"
    method_colors = {}  # Dictionary to store assigned colors for methods
    line_styles = {}  # Dictionary to store assigned line styles for methods
    muon_outputs = []
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        if name.startswith("muon*-"):
            muon_outputs.append(output)  # Store the output for this method
        if name.startswith("muon*-") and name not in method_colors:
            method_colors[name] = color_list[color_index % len(color_list)]
            line_styles[name] = line_style_list[color_index % len(line_style_list)]
            color_index += 1

    outputs = muon_outputs
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

    # Michael: Temporarily resetting matplotlib settings to default so that latex doesn't
    # need to be used for plot formatting. Was giving me an error.
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Plot final loss vs learning rate
    plot_final_loss_vs_lr(outputs, method_colors, outfilename)
    plot_final_loss_vs_lr(outputs, method_colors, outfilename, val=True)
    # Plot loss
    initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0  # Default to 1.0 if not available
    upper_bound = initial_loss * 1.2  # Set upper bound to 20% above the initial loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_data(ax, outputs, config['training_params']['num_epochs'], 'losses', 'Loss', method_colors, line_styles, lr_ranges, get_alpha_from_lr)
    lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    ax.set_ylim(lower_bound, upper_bound)  # Set the upper bound
    ax.legend(loc='upper right', fontsize=10)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)

    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)



