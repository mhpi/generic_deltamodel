import matplotlib.pyplot as plt
import numpy as np


def plot_cdf(
    metrics: dict[str, list[float]],
    metric_names: list[str],
    model_labels: list[str] = None,
    title: str = "CDF Comparison",
    xlabel: str = "Metric Value",
    figsize: tuple = None,
    xbounds: tuple = None,
    ybounds: tuple = None,
    show_arrow: bool = False,
    dpi: int = 100,
    fontsize: int = 12,
    ticksize: int = 10,
):
    """
    Plots cumulative distribution function (CDF) for specified metric(s) for
    one or many models.

    Parameters
    ----------
    metrics
        Dictionary containing metrics data for each model.
    metric_names
        List of metric names to plot.
    model_labels
        List of labels for each model. Default is None.
    title
        Title of the plot. Default is "CDF Comparison".
    xlabel
        Label for the x-axis. Default is "Metric Value".
    figsize
        Size of the figure. Default is None and plt will autosize.
    xbounds
        Bounds for the x-axis. Default is None and plt will autoscale.
    ybounds
        Bounds for the y-axis. Default is None and plt will autoscale.
    show_arrow
        Show an arrow pointing in the direction of better performance. Default
        is False.
    dpi
        Resolution of the figure. Default is 100.
    fontsize
        Font size for the plot. Default is 12.
    ticksize
        Font size for the ticks. Default is 10.
    """
    if not metrics or not metric_names:
        print("Error: No metrics data or metric names provided.")
        return

    # Ensure model_labels matches the number of models
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(metrics))]
    elif len(model_labels) != len(metrics):
        print("Error: Number of model labels must match the number of models.")
        return

    # Create the plot
    if figsize:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=(10, 6), dpi=dpi)

    for metric_name in metric_names:
        for i, model_data in enumerate(metrics):
            if metric_name in model_data and isinstance(model_data[metric_name], list):
                values = np.array(model_data[metric_name])
                sorted_values = np.sort(values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

                # Plot the CDF
                plt.plot(sorted_values, cdf, label=f"{model_labels[i]} - {metric_name.capitalize()}")
            else:
                print(f"Warning: Metric '{metric_name}' not found or invalid in model {i+1}.")

    # Add labels, legend, and grid
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("CDF", fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    if len(model_labels) > 1:
        # No need to show legend if only one model.
        plt.legend(loc="best")
    
    plt.grid(True)
    
    if xbounds:
        plt.xlim(xbounds)
    if ybounds:
        plt.ylim(ybounds)
    
    # Add an arrow annotation if requested
    if show_arrow:
        # Get the axis limits
        xlim = plt.xlim()
        ylim = plt.ylim()

        arrow_x_start = xlim[0] + 0.15 * (xlim[1] - xlim[0])  # Start slightly to the right of the y-axis
        arrow_x_end = xlim[0] + 0.3 * (xlim[1] - xlim[0])    # End near the right edge
        arrow_y = ylim[0] + 0.25 * (ylim[1] - ylim[0])                # Keep it centered vertically

        # Add the arrow without text
        plt.annotate(
            "",  # No text
            xy=(arrow_x_end, arrow_y),
            xytext=(arrow_x_start, arrow_y),  # Keep the y-coordinate constant
            arrowprops={'facecolor':'red', 'arrowstyle':'->', 'lw':2},
            fontsize=fontsize,
        )
        
        plt.text(arrow_x_start - 0.1, arrow_y, "Better", fontsize=12, color='black', ha='left', va='center')
    
    # Show the plot
    plt.show()
