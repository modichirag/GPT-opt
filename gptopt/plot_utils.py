import numpy as np

def get_alpha_from_lr(lr, min_alpha=0.5, max_alpha=1.0, lr_range=None):
    """Calculate alpha transparency based on the base learning rate."""
    if lr_range and lr_range[0] == lr_range[1]:  # Single learning rate case
        return max_alpha
    return min_alpha + (max_alpha - min_alpha) * (lr - lr_range[0]) / (lr_range[1] - lr_range[0])

def percentage_of_epoch(output, field, num_epochs):
    """Calculate the percentage of epochs for a given field."""
    total_iterations = len(output[field])
    percentages = [i / total_iterations * num_epochs for i in range(total_iterations)]
    return percentages

def get_lr_and_name(output):
    lr = float(output['config']['optimizer_params']['args']['lr'])
    alg_name = output['config']['optimizer_params']['name']
    if alg_name == 'muon':
        polar_method = output['config']['optimizer_params']['args']['polar_method']
        translate = {"Keller": "Jordan", "Jiacheng": "You"}
        if polar_method in translate: polar_method = translate[polar_method]
        name = f"muon-{polar_method}"
    else:
        name = str(alg_name)
    return name, lr

def plot_data(ax, outputs, num_epochs, field, ylabel, colormap, linestylemap, lr_ranges, alpha_func, zorder_func=None, time=False):
    """Generalized function to plot data."""
    plotted_methods = set()
    for output in outputs:
        name, lr = get_lr_and_name(output)
        lr = float(lr)
        alpha = alpha_func(lr, lr_range=lr_ranges[name])

        label = None
        if name not in plotted_methods:
            if lr_ranges[name][0] == lr_ranges[name][1]:  # Single learning rate
                label = f"{name} lr={lr_ranges[name][0]:.4f}"
            else:  # Range of learning rates
                label = f"{name} lr in [{lr_ranges[name][0]:.4f}, {lr_ranges[name][1]:.4f}]"

        zorder = zorder_func(name) if zorder_func else 1
        scale = 1
        if time:
            scale = np.sum(output['logs']['step_times'])
        x_values = percentage_of_epoch(output['logs'], field, num_epochs=num_epochs)
        x_values = [x * scale for x in x_values]
        ax.plot(x_values,
                output['logs'][field],
                label=label,
                color=colormap.get(name),
                linewidth=2,
                linestyle=linestylemap.get(name),
                alpha=alpha,
                zorder=zorder)
        plotted_methods.add(name)
    if time:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

def plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, alpha_func):
        """Generalized function to plot step_size_list and learning_rates."""
        plotted_methods = set()
        for output in outputs:
            if 'step_size_list' not in output['logs'] or 'learning_rates' not in output['logs']:
                continue

            name, lr = get_lr_and_name(output)
            lr = float(lr)
            alpha = alpha_func(lr, lr_range=lr_ranges[name])

            label = None
            if name not in plotted_methods:
                if lr_ranges[name][0] == lr_ranges[name][1]:
                    label = f"{name} lr={lr_ranges[name][0]:.1e}"
                else:
                    label = f"{name} lr in [{lr_ranges[name][0]:.1e}, {lr_ranges[name][1]:.1e}]"

            ax.plot(range(len(output['logs']['step_size_list'])),
                    output['logs']['step_size_list'],
                    label=label,
                    color=colormap[name],
                    linewidth=2,
                    linestyle=linestylemap[name],
                    alpha=alpha)

            ax.plot(range(len(output['logs']['learning_rates'])),
                    output['logs']['learning_rates'],
                    color=colormap[name],
                    linewidth=1.5,
                    linestyle='--',
                    alpha=alpha)

            plotted_methods.add(name)

        return plotted_methods


## Plotting related functions
def smoothen_curve_batch(data, num_points):
    smooth_data =[data[0]]
    t =0
    data_av = 0.0
    total_iterations = len(data)
    av_interval = max(1, total_iterations // num_points)

    for count, item in enumerate(data, start=0): 
        data_av = data_av*t/(t+1) + item*(1/(t+1))
        t = t+1
        if count % av_interval == 0:
            smooth_data.append(data_av)
            data_av =0.0
            t=0.0
    return smooth_data

def smoothen_curve_exp(data, num_points, beta=0.05):
    smooth_data =[data[0]]
    data_av = data[0]
    total_iterations = len(data)
    av_interval = max(1, total_iterations // num_points)
    for count, item in enumerate(data, start=0): 
        if np.isnan(item):
            continue
        data_av = (1-beta)*data_av + beta*item
        if count % av_interval == 0:
            smooth_data.append(data_av)
    return smooth_data

def smoothen_dict(dict, num_points, beta= 0.05):
    for key in dict.keys():
        if key == 'losses':
            dict[key] = smoothen_curve_exp(dict[key], num_points, beta = beta)

        """
        Michael: Temporarily removing smoothing of step_size_list. smoothen_curve_exp is
        breaking for Momo-Adam because it has two step sizes at each iteration.
        """
        # elif key == 'step_size_list':
        #     dict[key] = smoothen_curve_exp(dict[key], len(dict[key]), beta = beta)

        # dict[key] = smoothen_curve(dict[key], num_points)


