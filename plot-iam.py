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

TUNED_FIGSIZE = (4.14, 3.15)
LR_FIGSIZE = (4.05, 3.12)

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
                output['_source_file'] = file_name
                outputs.append(output)
    return outputs


EXCLUDED_METHODS = {
    'sgd-schedulep',
    'adamw-schedulep',
}


TEMP_EXCLUDED_METHODS = {
    # Add methods here to temporarily hide them from non-sensitivity plots.
}


IAM_METHODS = {'iams', 'iams-adam'}


method_labelmap = {
    'sgd-m-constant': 'SGD (constant)',
    'sgd-m-schedule': 'SGD (schedule)',
    'adamw-constant': 'AdamW (constant)',
    'adamw-schedule': 'AdamW (schedule)',
    'sgd-m': 'SGD',
    'adamw': 'AdamW',
    'iams': 'IAM',
    'iams-adam': 'IAM-Adam',
    'teacher': 'Teacher',
    'sgd-schedulep': r'SF-SPS$_+$',
    'adamw-schedulep': r'SF-Adam-SPS$_+$',
    'sgd-schedulefree': 'SF-SGD',
    'adamw-schedulefree': 'SF-Adam',
}


def split_output_name(output):
    return output['name'].split('-lr-')


def schedule_suffix_from_source(output):
    source_file = output.get('_source_file', '')
    if '-constant-' in source_file:
        return 'constant'
    if '-warm-up-cosine-' in source_file:
        return 'schedule'
    return None


def normalized_method_name(output):
    name, _ = split_output_name(output)
    schedule_suffix = schedule_suffix_from_source(output)
    if name in {'sgd-m', 'adamw'} and schedule_suffix:
        return f'{name}-{schedule_suffix}'
    return name


def prepare_outputs_for_plotting(outputs, include_temp_excluded=False):
    plot_outputs = []
    for output in outputs:
        name, lr = split_output_name(output)
        if name in EXCLUDED_METHODS:
            continue
        if not include_temp_excluded and name in TEMP_EXCLUDED_METHODS:
            continue

        plot_output = dict(output)
        plot_output['name'] = f'{normalized_method_name(output)}-lr-{lr}'
        plot_outputs.append(plot_output)
    return plot_outputs


def select_tuned_outputs(outputs, field):
    tuned_methods = {}
    for output in outputs:
        if field not in output:
            continue
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
    return tuned_methods


def replace_iam_learning_rates_with_step_sizes(outputs):
    plot_outputs = []
    for output in outputs:
        name, _ = output['name'].split('-lr-')
        plot_output = dict(output)
        if name in IAM_METHODS and 'step_size_list' in output:
            plot_output['learning_rates'] = output['step_size_list']
        plot_outputs.append(plot_output)
    return plot_outputs


def plot_teacher_curve(ax, outputs, reference_output, field, num_epochs, colormap, linestylemap, wallclock=False):
    teacher_output = next((output for output in outputs if 'teach_losses' in output), None)
    if teacher_output is None or reference_output is None:
        return

    teacher_loss = np.mean(teacher_output['teach_losses'])
    if wallclock:
        if "step_times" not in reference_output:
            return
        assert len(reference_output["step_times"]) % len(reference_output[field]) == 0
        step_factor = len(reference_output["step_times"]) // len(reference_output[field])
        step_times = np.array(reference_output["step_times"])
        xs = np.cumsum(np.sum(step_times.reshape((len(reference_output[field]), step_factor)), axis=1))
    else:
        xs = percentage_of_epoch(reference_output, field, num_epochs=num_epochs)

    ax.plot(xs,
            teacher_loss * np.ones(len(xs)),
            label='teacher',
            color=colormap.get('teacher', 'k'),
            linewidth=3.0,
            linestyle=linestylemap.get('teacher', '--'))


def format_method_label(label):
    for name, display_name in method_labelmap.items():
        if label == name or label.startswith(name + ' '):
            return display_name + label[len(name):]
    return label


def legend_with_method_labels(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [format_method_label(label) for label in labels], *args, **kwargs)


def use_scientific_yticks(ax):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


def plot_final_loss_vs_lr(outputs, colormap, linestylemap, outfilename, val=False):
    fig, ax = plt.subplots(figsize=(6.9, 4))
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
    if val:
        legend_with_method_labels(ax, loc='upper right')
    ax.grid(axis='both', lw=0.4, ls='--', zorder=0)
    use_scientific_yticks(ax)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    fig.savefig(plotfile, format='pdf', bbox_inches='tight')


def plot_tuned_curves(outputs, colormap, linestylemap, outfilename, num_epochs, wallclock=False, val=False):
    fig, ax = plt.subplots(figsize=TUNED_FIGSIZE)

    field = 'val_losses' if val else 'losses'
    tuned_methods = select_tuned_outputs(outputs, field)
    print("Best Validation losses:" if val else "Best losses:")
    for name in tuned_methods:
        print(f"{name}: {tuned_methods[name]['best_loss']} at lr {tuned_methods[name]['best_lr']}")

    tuned_outputs = [tuned_methods[name]['outputs'] for name in tuned_methods]
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, field, 'Loss', colormap, linestylemap,
              lr_ranges, get_alpha_from_lr, wallclock=wallclock)
    if tuned_outputs:
        plot_teacher_curve(ax, outputs, tuned_outputs[0], field, num_epochs,
                           colormap, linestylemap, wallclock=wallclock)
    for line in ax.lines:
        line.set_linewidth(3.0)
    # Temporarily hidden for tuned plots.
    # legend_with_method_labels(ax, loc='upper right')
    if val and not wallclock:
        xmax = max((len(output[field]) - 1) / len(output[field]) * num_epochs for output in tuned_outputs)
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.linspace(0, xmax, 6))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, num_epochs, 6)])
    use_scientific_yticks(ax)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    suffix = "_tuned"
    if wallclock:
        suffix += "_wallclock"
    if val:
        suffix += "_val"
    fig.savefig("figures/" + outfilename + suffix + '.pdf', format='pdf', bbox_inches='tight')


def plot_tuned_learning_rates(outputs, colormap, linestylemap, outfilename, num_epochs,
                              method_subset, suffix, val=False):
    fig, ax = plt.subplots(figsize=LR_FIGSIZE)

    field = 'val_losses' if val else 'losses'
    tuned_methods = select_tuned_outputs(outputs, field)

    tuned_methods = {
        name: data
        for name, data in tuned_methods.items()
        if name in method_subset
    }
    tuned_outputs = replace_iam_learning_rates_with_step_sizes(
        [tuned_methods[name]['outputs'] for name in tuned_methods]
    )
    lr_ranges = {name: [tuned_methods[name]['best_lr']] * 2 for name in tuned_methods}
    plot_data(ax, tuned_outputs, num_epochs, 'learning_rates', 'Learning rate',
              colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    # Temporarily hidden for tuned LR plots.
    # legend_with_method_labels(ax, loc='upper right')
    use_scientific_yticks(ax)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig("figures/" + outfilename + suffix + '.pdf', format='pdf', bbox_inches='tight')


def main(config_file=None):
    default_config = get_default_config()
    if config_file:
        config = load_config(default_config, config_file)
    outfilename = config_file.replace("configs/", "").replace('.yaml', '').replace('.yml', '')
    output_root = "gptopt/outputs"
    output_dirs = [
        os.path.join(output_root, name)
        for name in os.listdir(output_root)
        if name == outfilename or (
            name.startswith(outfilename + '-')
            and name[len(outfilename) + 1:].isdigit()
        )
    ]
    output_dirs.sort(key=lambda path: (len(os.path.basename(path)), os.path.basename(path)))
    raw_outputs = []
    for output_dir in output_dirs:
        raw_outputs.extend(load_outputs(output_dir))

    print(f"Loaded {len(raw_outputs)} outputs from {output_dirs}")
    sensitivity_outputs = prepare_outputs_for_plotting(raw_outputs, include_temp_excluded=True)
    outputs = prepare_outputs_for_plotting(raw_outputs)
    print(f"Plotting {len(sensitivity_outputs)} sensitivity outputs after method filtering")
    print(f"Plotting {len(outputs)} outputs after temporary method filtering")

    for output in sensitivity_outputs:
        smoothen_dict(output, num_points=None, beta=0.05)
    for output in outputs:
        smoothen_dict(output, num_points=None, beta=0.05)

    colormap = {
        'sgd-m-constant': '#B3CBB9',
        'sgd-m-schedule': '#B3CBB9',
        'adamw-constant': '#FF6B35',
        'adamw-schedule': '#FF6B35',
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
        'sgd-m-constant': None,
        'sgd-m-schedule': '--',
        'sgd-sch': '--',
        'teacher': '--',
        'iams-adam': None,
        'adam': None,
        'adamw': '--',
        'adamw-constant': None,
        'adamw-schedule': '--',
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

    plot_final_loss_vs_lr(sensitivity_outputs, colormap, linestylemap, outfilename)
    plot_final_loss_vs_lr(sensitivity_outputs, colormap, linestylemap, outfilename, val=True)

    initial_loss = outputs[0]['losses'][0] if outputs and 'losses' in outputs[0] else 1.0
    upper_bound = initial_loss * 1.2
    fig, ax = plt.subplots(figsize=(4.83, 3.2))
    plot_data(ax, outputs,  config['training_params']['num_epochs'], 'losses', 'Loss',
              colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    lower_bound = min(min(output['losses']) for output in outputs if 'losses' in output)
    lower_bound *= 0.95
    ax.set_ylim(lower_bound, upper_bound)
    legend_with_method_labels(ax, loc='upper right')
    use_scientific_yticks(ax)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '.pdf', format='pdf', bbox_inches='tight')

    for method_subset in [['sgd-m-constant', 'sgd-m-schedule', 'iams'],
                          ['adamw-constant', 'adamw-schedule', 'iams-adam']]:
        fig, ax = plt.subplots(figsize=LR_FIGSIZE)
        subset_outputs = [output for output in outputs if output['name'].split('-lr-')[0] in method_subset]
        subset_outputs = replace_iam_learning_rates_with_step_sizes(subset_outputs)
        plot_data(ax, subset_outputs, config['training_params']['num_epochs'], 'learning_rates',
                  'Learning rate', colormap, linestylemap, lr_ranges, get_alpha_from_lr)
        legend_with_method_labels(ax, loc='upper right')
        use_scientific_yticks(ax)
        fig.subplots_adjust(top=0.935, bottom=0.03, left=0.155, right=0.99)
        name = '-lr' if any(method.startswith('sgd-m') for method in method_subset) else '-lr-adam'
        fig.savefig('figures/' + outfilename + name + '.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=LR_FIGSIZE)
    plotted_methods = plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, get_alpha_from_lr)
    handles, labels = ax.get_legend_handles_labels()
    legend_handles = [copy.copy(handle) for handle in handles]
    for handle in legend_handles:
        handle.set_alpha(1.0)
    ax.legend(legend_handles, [format_method_label(label) for label in labels], loc='upper right')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    use_scientific_yticks(ax)
    fig.subplots_adjust(top=0.99, bottom=0.155, left=0.12, right=0.99)
    fig.savefig('figures/' + outfilename + '-step_size-.pdf', format='pdf', bbox_inches='tight')

    plot_tuned_curves(outputs, colormap, linestylemap, outfilename,
                      config['training_params']['num_epochs'], wallclock=False, val=False)
    plot_tuned_curves(outputs, colormap, linestylemap, outfilename,
                      config['training_params']['num_epochs'], wallclock=False, val=True)
    plot_tuned_learning_rates(outputs, colormap, linestylemap, outfilename,
                              config['training_params']['num_epochs'],
                              ['sgd-m-constant', 'sgd-m-schedule', 'iams', 'sgd-schedulefree'],
                              '_tuned_lr',
                              val=False)
    plot_tuned_learning_rates(outputs, colormap, linestylemap, outfilename,
                              config['training_params']['num_epochs'],
                              ['adamw-constant', 'adamw-schedule', 'iams-adam', 'adamw-schedulefree'],
                              '_tuned_adam_lr',
                              val=False)
    plot_tuned_learning_rates(outputs, colormap, linestylemap, outfilename,
                              config['training_params']['num_epochs'],
                              ['sgd-m-constant', 'sgd-m-schedule', 'iams', 'sgd-schedulefree'],
                              '_tuned_val_lr',
                              val=True)
    plot_tuned_learning_rates(outputs, colormap, linestylemap, outfilename,
                              config['training_params']['num_epochs'],
                              ['adamw-constant', 'adamw-schedule', 'iams-adam', 'adamw-schedulefree'],
                              '_tuned_val_adam_lr',
                              val=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting gpt_distill outputs.')
    parser.add_argument('config', type=str, nargs='?', help='Path to config file', default=None)
    args = parser.parse_args()
    if args.config:
        print(f"Loading configuration from {args.config}")
    else:
        print("No config file provided, using default settings.")
    main(args.config)
