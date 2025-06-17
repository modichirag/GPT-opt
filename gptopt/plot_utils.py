import numpy as np

def get_alpha_from_lr(lr, min_alpha=0.3, max_alpha=1.0, lr_range=None):
    """Calculate alpha transparency based on the base learning rate."""
    if lr_range and lr_range[0] == lr_range[1]:  # Single learning rate case
        return max_alpha
    return min_alpha + (max_alpha - min_alpha) * (lr - lr_range[0]) / (lr_range[1] - lr_range[0])

def percentage_of_epoch(output, field, num_epochs):
    """Calculate the percentage of epochs for a given field."""
    total_iterations = len(output[field])
    percentages = [i / total_iterations * num_epochs for i in range(total_iterations)]
    return percentages

def plot_data(ax, outputs, num_epochs, field, ylabel, colormap, linestylemap, lr_ranges, alpha_func, zorder_func=None, wallclock=False):
    """Generalized function to plot data."""
    plotted_methods = set()
    for output in outputs:
        name, lr = output['name'].split('-lr-')
        lr = float(lr)
        alpha = alpha_func(lr, lr_range=lr_ranges[name])

        label = None
        if name not in plotted_methods:
            if lr_ranges[name][0] == lr_ranges[name][1]:  # Single learning rate
                label = f"{name} lr={lr_ranges[name][0]:.4f}"
            else:  # Range of learning rates
                label = f"{name} lr in [{lr_ranges[name][0]:.4f}, {lr_ranges[name][1]:.4f}]"

        zorder = zorder_func(name) if zorder_func else 1

        if wallclock:
            assert len(output["step_times"]) % len(output[field]) == 0
            step_factor = len(output["step_times"]) // len(output[field])
            step_times = np.array(output["step_times"])
            step_times = np.sum(step_times.reshape((len(output[field]), step_factor)), axis=1)
            xs = np.cumsum(step_times)
        else:
            xs = percentage_of_epoch(output, field, num_epochs=num_epochs)

        ax.plot(xs,
                output[field],
                label=label,
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                alpha=alpha,
                zorder=zorder)
        plotted_methods.add(name)

    xlabel = "Seconds" if wallclock else "Epochs"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='both', lw=0.2, ls='--', zorder=0)

def plot_step_size_and_lr(ax, outputs, colormap, linestylemap, lr_ranges, alpha_func):
        """Generalized function to plot step_size_list and learning_rates."""
        plotted_methods = set()
        for output in outputs:
            if 'step_size_list' not in output or 'learning_rates' not in output:
                continue

            name, lr = output['name'].split('-lr-')
            lr = float(lr)
            alpha = alpha_func(lr, lr_range=lr_ranges[name])

            label = None
            if name not in plotted_methods:
                if lr_ranges[name][0] == lr_ranges[name][1]:
                    label = f"{name} lr={lr_ranges[name][0]:.1e}"
                else:
                    label = f"{name} lr in [{lr_ranges[name][0]:.1e}, {lr_ranges[name][1]:.1e}]"

            ax.plot(range(len(output['step_size_list'])),
                    output['step_size_list'],
                    label=label,
                    color=colormap[name],
                    linewidth=2,
                    linestyle=linestylemap[name],
                    alpha=alpha)

            ax.plot(range(len(output['learning_rates'])),
                    output['learning_rates'],
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

def smoothen_curve_exp(data, num_points=None, beta=0.05):
    smooth_data =[data[0]]
    data_av = data[0]
    total_iterations = len(data)
    if num_points is None:
        num_points = total_iterations
    av_interval = max(1, total_iterations // num_points)
    for count, item in enumerate(data, start=0): 
        if np.isnan(item):
            continue
        data_av = (1-beta)*data_av + beta*item
        if count % av_interval == 0:
            smooth_data.append(data_av)
    return smooth_data

def smoothen_dict(dict, num_points=None, beta= 0.05):
    for key in dict.keys():
        if key == 'losses':
            dict[key] = smoothen_curve_exp(dict[key], num_points=None, beta = beta)
        elif key == 'step_times':
            if num_points is not None:
                raise NotImplementedError("Plotting by wallclock time is not compatible with changing the number of plotted points through smoothing.")

        """
        Michael: Temporarily removing smoothing of step_size_list. smoothen_curve_exp is
        breaking for Momo-Adam because it has two step sizes at each iteration.
        """
        # elif key == 'step_size_list':
        #     dict[key] = smoothen_curve_exp(dict[key], len(dict[key]), beta = beta)

        # dict[key] = smoothen_curve(dict[key], num_points)


