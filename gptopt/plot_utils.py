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

def plot_data(ax, outputs, num_epochs, field, ylabel, colormap, linestylemap, lr_ranges, alpha_func, zorder_func=None):
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
        ax.plot(percentage_of_epoch(output, field, num_epochs=num_epochs),
                output[field],
                label=label,
                color=colormap[name],
                linewidth=2,
                linestyle=linestylemap[name],
                alpha=alpha,
                zorder=zorder)
        plotted_methods.add(name)

    ax.set_xlabel('Epoch')
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