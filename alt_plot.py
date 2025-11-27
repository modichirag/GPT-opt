import argparse
from collections import defaultdict
import yaml
import matplotlib.pyplot as plt
from gptopt.utils import get_default_config, load_config
from gptopt.plot_utils import get_alpha_from_lr, plot_data, plot_step_size_and_lr, smoothen_dict, get_lr_and_name
import copy
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from math import inf


colormap = {
    'sgd-m': '#B3CBB9',
    'sgd-sch': '#B3CBB9',
    'adam': '#00518F',
    'adamw': '#00518F',  # Oragne'#FF6B35',
    'adam-sch': '#FF6B35',
    'momo': '#61ACE5',
    'muon-polarexpress': 'k',
    'muon-You': '#8A2BE2',  # Added a new color for "muon" (blue-violet)
    'muon-Jordan': '#FF0000',
}

linestylemap = {
    'momo': None,
    'sgd-m': None,
    'sgd-sch': '--',
    'muon-polarexpress': None,
    'adam': None,
    'adamw': None,
    'adam-sch': '--',
    'muon-You': ':',
    'muon-Jordan': '-.',
}


def dict2frozen(a_dict):
    return frozenset(
        (k, dict2frozen(v)) if isinstance(v, dict) 
        else (k, v) for k, v in a_dict.items()
    )


def frozen2dict(a_frozen):
    result = {}
    for k, v in a_frozen:
        if isinstance(v, frozenset):
            result[k] = frozen2dict(v)
        else:
            result[k] = v
    return result


def config_key(full_config):
    # Include everything in config except the optimizer params and the logging_params
    stripped_config = {
        k: v for k, v in full_config.items()
        if k not in ('optimizer_params', 'logging_params')
    }
    # But do include weight_decay, even though its an optimizer param
    stripped_config['weight_decay'] = full_config.get('optimizer_params', {}).get('args', {}).get('weight_decay', None)
    # and we do exclude training_data.training_params.val_tokens_processed, which used to be part of the config but didn't do anything
    if 'val_tokens_processed' in stripped_config.get('training_data', {}).get('training_params', {}):
        del stripped_config['training_data']['training_params']['val_tokens_processed']
    return dict2frozen(stripped_config)


def optimizer_config(full_config):
    # MUST deepcopy since we will modify args
    args = copy.deepcopy(full_config.get('optimizer_params', {}).get('args', {}))
    polar_args = args.pop('polar_args', {})
    polar_args = {f"polar_args.{k}": v for k, v in polar_args.items()}
    c = dict(name=full_config.get('optimizer_params', {}).get('name'), **args, **polar_args)
    if 'weight_decay' in c:
        del c['weight_decay']
    return c


def load_output_folder(experiment_results_folder):
    outputs = []
    for root, _, files in os.walk(experiment_results_folder, followlinks=True):
        for file_name in files:
            if file_name.startswith("logs") and file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    logs = json.load(file)
                with open(os.path.join(root, ".hydra/config.yaml"), 'r') as file:
                    config = yaml.safe_load(file)
                outputs.append(dict(config=config, logs=logs))
    return outputs


def make_summary_df(run_data):
    df = pd.DataFrame([
        dict(
            **optimizer_config(run_datum['config']),
            final_loss=run_datum['logs']['losses'][-1],
            final_val_loss=run_datum['logs']['val_losses'][-1],
            min_loss=min(run_datum['logs']['losses']),
            min_val_loss=min(run_datum['logs']['val_losses']),
            median_time=pd.Series(run_datum['logs']['step_times']).median(),
        )
        for run_datum in run_data
    ])
    df.rename(columns={
        "name": "Optimizer",
        "lr": "Learning Rate",
        "ns_steps": "Iterations",
        "polar_method": "Polar Method",
        "polar_args.svd_cutoff": "Cutoff",
        "polar_args.svd_reverse": "Reverse",
        "final_loss": "Final Training Loss",
        "final_val_loss": "Final Validation Loss",
        "median_time": "Median Training Step Time (s)",
    }, inplace=True)
    df["Polar Method"] = df["Polar Method"].map({"polarexpress": "PolarExpress", "svd-exact": "SVD"}.get)
    df.sort_values(by=['Optimizer', 'Polar Method', 'Iterations', 'Learning Rate'], inplace=True)
    return df


def loss_curve(selected_run_data, outfolder, metric="val_losses"):
    outfolder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=True)
    sorted_ns_steps = sorted(int(run_datum['config']['optimizer_params']['args']['ns_steps']) for run_datum in selected_run_data)
    polar_colors = sns.color_palette("rocket", n_colors=len(sorted_ns_steps))
    for run_datum in selected_run_data:
        opt_conf = optimizer_config(run_datum['config'])
        y = run_datum["logs"][metric]
        issvd = opt_conf["polar_method"] == "svd-exact"
        ax.plot(
            np.linspace(0, run_datum['config']["training_data"]["training_params"]["num_epochs"], num=len(y)),
            y,
            color='g' if issvd else polar_colors[sorted_ns_steps.index(opt_conf['ns_steps'])],
            label="SVD" if issvd else f"PolarExpress {opt_conf['ns_steps']}",
            linestyle=':' if issvd else '-',
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss")
    ax.legend()
    fig.savefig(outfolder / f"{metric}.pdf", bbox_inches='tight')


def final_loss_barplot(run_data, outfolder):
    outfolder.mkdir(parents=True, exist_ok=True)
    df = make_summary_df(run_data)
    def label(row):
        if row["Polar Method"] == "SVD":
            return "SVD"
        else:
            return row["Iterations"]
    df['Iterations of Polar Express'] = df.apply(label, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), squeeze=True)
    sns.barplot(data=df, x='Iterations of Polar Express', y="Final Validation Loss", hue=df["Polar Method"], legend=True, ax=axes[0])
    sns.barplot(data=df, x='Iterations of Polar Express', y="Median Training Step Time (s)", hue=df["Polar Method"], legend=False, ax=axes[1])

    lower, upper = df["Final Validation Loss"].min(), df["Final Validation Loss"].max()
    yrange = upper - lower
    axes[0].set_ylim(lower - 0.2 * yrange, upper + 0.3 * yrange)
    fig.savefig(outfolder / "compare_ns_steps.pdf", bbox_inches='tight')


def main(experiment_name, figures_dir):
    results_folder = f"outputs/hydra-results/{experiment_name}"
    all_runs = load_output_folder(results_folder)
    print("Total num experiments:", len(all_runs))
    runs_by_group = defaultdict(list)
    for run_data in all_runs:
        key = config_key(run_data['config'])
        runs_by_group[key].append(run_data)
    for group_ix, (key, run_data) in enumerate(runs_by_group.items()):
        print(f"Group {group_ix}: {len(run_data)} runs")
        group_out_path = Path(f"{figures_dir}/{experiment_name}/group{group_ix}/")
        group_out_path.mkdir(parents=True, exist_ok=True)
        with open(group_out_path / "common_config.yaml", 'w') as f:
            yaml.safe_dump(frozen2dict(key), f)
        sum_df = make_summary_df(run_data)
        print(list(sum_df))
        print(sum_df)
        final_loss_barplot(run_data, group_out_path)
        loss_curve(run_data, group_out_path)


def cutoff_barplot(run_data, outfolder):
    outfolder.mkdir(parents=True, exist_ok=True)
    df = make_summary_df(run_data)
    true_polar_label = r"$\sigma \mapsto 1$ (true polar)"
    reverse_label = r"$\sigma \mapsto \begin{cases}1 & \sigma > \gamma \sigma_{\max} \\ -1 & \sigma < \gamma \sigma_{\max}\end{cases}$"
    zero_label = r"$\sigma \mapsto \begin{cases}1 & \sigma > \gamma \sigma_{\max} \\ 0 & \sigma < \gamma \sigma_{\max}\end{cases}$"
    def group_label(row):
        if row["Reverse"] == True:
            return reverse_label
        elif row["Cutoff"] == 0 or pd.isna(row["Cutoff"]) or row["Cutoff"] is None:
            return true_polar_label
        else:
            return zero_label
    df["Cutoff ($\gamma$)"] = df["Cutoff"].apply(lambda x: 0 if x is None else x).astype(float)
    df["Function"] = df.apply(group_label, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(5, 3), sharex=True, squeeze=True)
    sns.lineplot(data=df[df["Function"] != true_polar_label], x="Cutoff ($\gamma$)", y="Final Validation Loss", hue="Function", style="Function", markers=True, legend=True, ax=ax[0])
    for y in df.loc[df["Function"] == true_polar_label, "Final Validation Loss"]:
        ax[0].axhline(y=y, color='tab:green', linestyle=':', label=true_polar_label)
    ax[0].legend(title="Function")
    plt.xscale('log')
    # lower, upper = df["Final Validation Loss"].min(), df["Final Validation Loss"].max()
    # yrange = upper - lower
    # ax.set_ylim(lower - 0.2 * yrange, upper + 0.3 * yrange)
    fig.savefig(outfolder / "compare_ns_steps.pdf", bbox_inches='tight')

    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3), squeeze=True)
    colors = {reverse_label: 'tab:blue', zero_label: 'tab:orange', true_polar_label: 'tab:green'}
    linestyles = {reverse_label: '-', zero_label: '--', true_polar_label: ':'}
    # this is kind of a hack. I should really construct a dataframe with all the data and use seaborn
    for run_datum, (_, row) in zip(run_data, df.iterrows()):
        y = run_datum["logs"]["val_losses"]
        ax2.plot(
            np.linspace(0, run_datum['config']["training_data"]["training_params"]["num_epochs"], num=len(y)),
            y,
            color=colors[row["Function"]],
            label=row["Function"],
            linestyle=linestyles[row["Function"]],
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss")
    ax.legend()
    fig2.savefig(outfolder / "val_losses.pdf", bbox_inches='tight')


def main2(experiment_name, figures_dir):
    results_folder = f"outputs/hydra-results/{experiment_name}"
    all_runs = load_output_folder(results_folder)
    print("Total num experiments:", len(all_runs))
    runs_by_group = defaultdict(list)
    for run_data in all_runs:
        key = config_key(run_data['config'])
        runs_by_group[key].append(run_data)
    for group_ix, (key, run_data) in enumerate(runs_by_group.items()):
        print(f"Group {group_ix}: {len(run_data)} runs")
        group_out_path = Path(f"{figures_dir}/{experiment_name}/group{group_ix}/")
        group_out_path.mkdir(parents=True, exist_ok=True)
        with open(group_out_path / "common_config.yaml", 'w') as f:
            yaml.safe_dump(frozen2dict(key), f)
        sum_df = make_summary_df(run_data)
        print(list(sum_df))
        print(sum_df)
        cutoff_barplot(run_data, group_out_path)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('text', usetex=True)
    plt.rc('legend', fontsize=10)

    parser = argparse.ArgumentParser(description='Plotting outputs.')
    # parser.add_argument('--experiment_name', type=str, nargs='?', help='Path to results folder', default="ns-steps")
    parser.add_argument('--figures_dir', type=str, nargs='?', help='Path to results folder', default="alt_figures")
    args = parser.parse_args()

    main(experiment_name="ns-steps", figures_dir=args.figures_dir)
    main2(experiment_name="supression", figures_dir=args.figures_dir)
