import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gptopt.utils import smoothen_dict
from gptopt.utils import get_default_config, load_config, get_outputfile_from_configfile
import copy 
import json
import numpy as np 

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rc('text', usetex=True)
plt.rc('legend',fontsize=10) 

def main(config_file=None):

    default_config = get_default_config() 
    if config_file:
        config = load_config(default_config, config_file)

    outputfile = get_outputfile_from_configfile(config_file) 
    with open(outputfile, 'r') as file: outputs = json.load(file)

    for output in outputs: #Smoothing
        smoothen_dict(output, num_points=100)
    
    def percentage_of_epoch(output, field):
        total_iterations = len(output[field])
        percentages = [i /total_iterations   * config['training_params']['num_epochs'] for i in range(total_iterations)]
        return percentages
    
    colormap = {'sgd' : '#B3CBB9',
                'sgd-sch': '#B3CBB9',
                'adam': '#FF6B35',
                'adam-sch' : '#FF6B35',
                'iam' : '#61ACE5',
                'iam-adam': '#00518F',
                'teacher' : 'k',
    }
    linestylemap =  {'iam' : None,
                     'sgd' : None,
                     'sgd-sch': '--',
                     'teacher' : '--',  
                     'iam-adam': None,
                     'adam': None,
                     'adam-sch' : '--'
    }
    markermap =  {'iam' : None, 'sgd' : None, 'sgd-sch': None, 'teacher' : None,  "iam-adam": None, 'adam': None, 'adam-sch' : None}
    
    # Plot loss
    fig, ax = plt.subplots(figsize=(4, 3))
    for output in outputs:
        if 'iams' in output['name']:
            output['name'] = output['name'].replace('iams', 'iam')
        name  = output['name'].split('-lr-')[0]  
        if output['name'] == 'iam':
            # plot hline with average teacher loss
            ax.hlines(np.mean(output['teacher_losses']),
                     0, 1,
                     label='teacher',
                     color="black",
                     linewidth=1.5,
                     ls='--'
            )
        ax.plot(percentage_of_epoch(output, 'losses'),
                output['losses'],
                label=output['name'],
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                markersize=10,
                alpha=0.95,
                zorder= 3 if 'iam' in output['name'] else 1
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

    fig.subplots_adjust(top=0.99,
                        bottom=0.155,
                        left=0.12,
                        right=0.99,)
    fig.savefig('figures/' + config['name'] + '.pdf', format='pdf', bbox_inches='tight')
    
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.yaxis.get_major_formatter().set_scientific(True)
    #ax.yaxis.get_major_formatter().set_powerlimits((-1, 1))
    
    # Plot learning rates
    for method_subset in [['sgd', 'sgd-sch', 'iam'], ['adam', 'adam-sch', 'iam-adam']]:
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
        # ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-1, 1))

        fig.subplots_adjust(top=0.935,
                            bottom=0.03,
                            left=0.155,
                            right=0.99)
        name = 'figures/lr-' if 'sgd' in method_subset else 'figures/lr-adam-'
        fig.savefig(name + config['name'] + '.pdf', format='pdf', bbox_inches='tight')
    
    # Plot legend
    from matplotlib.lines import Line2D
    label_mapping = {'teacher' : 'teacher',
                'sgd' : r'$\tt SGD$ (constant)',
                'sgd-sch': r'$\tt SGD$ (schedule)',
                'adam': r'$\tt Adam$ (constant)',
                'adam-sch' : r'$\tt Adam$ (schedule)',
                'iam' : r'$\tt IAM$',
                'iam-adam': r'$\tt IAM-Adam$'
    }
    fig, ax = plt.subplots(figsize=(5.7,0.6))
    ax.axis('off')
    handles = list()
    labels = list()
    for k, v in colormap.items():
        handles.append(Line2D([0], [0],
                              c=v,
                              linestyle='-' if linestylemap[k] is None else linestylemap[k],
                              lw=2))
        labels.append(label_mapping[k])
    
        
    fig.legend(handles, 
               labels, 
               loc='center', 
               fontsize=11, 
               framealpha=0, 
               ncol=4,
               mode="expand")
    fig.savefig('figures/legend.pdf', format='pdf', bbox_inches='tight')



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
    


