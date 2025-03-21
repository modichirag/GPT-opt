import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.utils import smoothen_dict
from gptopt.utils import get_default_config, load_config, get_outputfile_from_configfile
import copy 
import json
import numpy as np 
import os

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend',fontsize=10) 

def main(config_file=None):

    default_config = get_default_config() 
    if config_file:
        config = load_config(default_config, config_file)
    outfilename = config_file.replace("configs/","").replace('.yaml','')
    output_dir = f"gptopt/outputs/{outfilename}"
    outputs = []

    # Load all individual output files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'r') as file:
                output = json.load(file)
                outputs.append(output)

    print(f"Loaded {len(outputs)} outputs from {output_dir}")

    for output in outputs: #Smoothing
        smoothen_dict(output, num_points=100)
    
    def percentage_of_epoch(output, field):
        total_iterations = len(output[field])
        percentages = [i /total_iterations   * config['training_params']['num_epochs'] for i in range(total_iterations)]
        return percentages
    
    colormap = {'sgd-m' : '#B3CBB9',
                'sgd-sch': '#B3CBB9',
                'adam': '#FF6B35',
                'adam-sch' : '#FF6B35',
                'momo' : '#61ACE5',
                'momo-adam': '#00518F',
                'teacher' : 'k',
    }
    linestylemap =  {'momo' : None,
                     'sgd-m' : None,
                     'sgd-sch': '--',
                     'teacher' : '--',  
                     'momo-adam': None,
                     'adam': None,
                     'adam-sch' : '--'
    }
    markermap =  {'momo' : None, 'sgd-m' : None, 'sgd-sch': None, 'teacher' : None,  "momo-adam": None, 'adam': None, 'adam-sch' : None}
    
    def get_alpha_from_lr(lr, min_alpha=0.3, max_alpha=1.0, lr_range=None):
        """Calculate alpha transparency based on the base learning rate."""
        if lr_range and lr_range[0] == lr_range[1]:  # Single learning rate case
            return max_alpha
        return min_alpha + (max_alpha - min_alpha) * (lr - lr_range[0]) / (lr_range[1] - lr_range[0])

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

    # Plot loss
    fig, ax = plt.subplots(figsize=(4, 3))
    plotted_methods = set()  # To track methods already added to the legend
    for output in outputs:
        name, lr = output['name'].split('-lr-')  # Split to get method name and learning rate
        lr = float(lr)  # Convert learning rate to float
        alpha = get_alpha_from_lr(lr, lr_range=lr_ranges[name])  # Calculate alpha transparency

        if output['name'] == 'momo':
            # plot hline with average teacher loss
            ax.hlines(output['lb'],
                     0, 1,
                     label='teacher' if 'teacher' not in plotted_methods else None,
                     color="black",
                     linewidth=1.5,
                     ls='--'
            )
            plotted_methods.add('teacher')

        label = None
        if name not in plotted_methods:
            if lr_ranges[name][0] == lr_ranges[name][1]:  # Single learning rate
                label = f"{name} lr={lr_ranges[name][0]:.4f}"
            else:  # Range of learning rates
                label = f"{name} lr in [{lr_ranges[name][0]:.4f}, {lr_ranges[name][1]:.4f}]"

        ax.plot(percentage_of_epoch(output, 'losses'),
                output['losses'],
                label=label,  # Add to legend only once
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                markersize=10,
                alpha=alpha,  # Set alpha transparency
                zorder=3 if 'momo' in output['name'] else 1
        )
        plotted_methods.add(name)  # Mark method as added to the legend


    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

    fig.subplots_adjust(top=0.99,
                        bottom=0.155,
                        left=0.12,
                        right=0.99,)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')
    
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.yaxis.get_major_formatter().set_scientific(True)
    #ax.yaxis.get_major_formatter().set_powerlimits((-1, 1))
    
    # Plot learning rates
    for method_subset in [['sgd-m', 'sgd-sch', 'momo'], ['adam', 'adam-sch', 'momo-adam']]:
        fig, ax = plt.subplots(figsize=(4, 3))
        for output in outputs:
            name  = output['name'].split('-lr-')[0]
            if name in method_subset:
                plt.plot(percentage_of_epoch(output, 'learning_rates'),
                        output['learning_rates'],
                        label=output['name'],
                        marker=markermap[name],
                        markevery =len(output['losses'])//4,
                        color=colormap[name],
                        linewidth=2,
                        linestyle=linestylemap[name],
                        markersize=10)
            else:
                continue
        
        ax.grid(axis='both', lw=0.2, ls='--', zorder=0)
        ax.set_xticklabels([])  # Remove x-ticks
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('Learning rate')
        ax.legend(loc='upper right', fontsize=10)
        # ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-1, 1))

        fig.subplots_adjust(top=0.935,
                            bottom=0.03,
                            left=0.155,
                            right=0.99)
        name = 'figures/lr-' if 'sgd-m' in method_subset else 'figures/lr-adam-'
        fig.savefig(name + outfilename + '.pdf', format='pdf', bbox_inches='tight')
# Plot step size lists
    fig, ax = plt.subplots(figsize=(4, 3))
    plotted_methods = set()  # To track methods already added to the legend
    for output in outputs:
        if 'step_size_list' not in output or 'learning_rates' not in output:
            continue  # Skip outputs without step_size_list or learning_rates
        
        name, lr = output['name'].split('-lr-')  # Split to get method name and learning rate
        lr = float(lr)  # Convert learning rate to float
        alpha = get_alpha_from_lr(lr, lr_range=lr_ranges[name])  # Calculate alpha transparency

        label = None
        if name not in plotted_methods:
            if lr_ranges[name][0] == lr_ranges[name][1]:  # Single learning rate
                label = f"{name} lr={lr_ranges[name][0]:.1e}"  # Use scientific notation
            else:  # Range of learning rates
                label = f"{name} lr in [{lr_ranges[name][0]:.1e}, {lr_ranges[name][1]:.1e}]"  # Use scientific notation

        # Plot step_size_list
        ax.plot(range(len(output['step_size_list'])),
                output['step_size_list'],
                label=label,  # Add to legend only once
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                markersize=10,
                alpha=alpha)  # Set alpha transparency

        # Plot learning_rates
        ax.plot(range(len(output['learning_rates'])),
                output['learning_rates'],
                color=colormap[name],
                linewidth=1.5,
                linestyle='--',
                alpha=alpha)  # Dashed line for learning_rates

        plotted_methods.add(name)  # Mark method as added to the legend

    # Update legend to use opaque colors without affecting plot transparency
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(handle) for handle in handles]  # Create copies for the legend
    for handle in legend_handles:
        handle.set_alpha(1.0)  # Set alpha to 1.0 for opaque colors in the legend
    ax.legend(legend_handles, labels, loc='upper right', fontsize=10)

    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

    fig.subplots_adjust(top=0.99,
                        bottom=0.155,
                        left=0.12,
                        right=0.99)
    fig.savefig('figures/step_size-' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    # Argument parser to optionally provide a config file
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    
    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)



