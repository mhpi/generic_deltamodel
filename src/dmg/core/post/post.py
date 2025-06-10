def print_metrics(
    metrics: dict[str, dict[str, float]],
    metric_names: str,
    mode: str = 'median',
    precision: int = 3,
) -> None:
    """
    Prints either the median or the mean ± standard deviation for the specified metrics,
    with a specified number of significant digits after the decimal point.

    Parameters
    ----------
    metrics
        Dictionary containing the metrics data.
    metric_names
        List of metric names to process.
    mode
        Either "median" or "mean_std". Defaults to "median".
    precision
        Number of significant digits after the decimal point. Defaults to 3.
    """
    if mode not in ["median", "mean_std"]:
        print("Error: Mode must be 'median' or 'mean_std'.")
        return

    if mode == "median":
        mode_name = "Median"
    else:
        mode_name = "Mean ± Std"

    print(f"{mode_name} of Metrics (Prec: {precision} digits):")
    print("-" * 40)
    for name in metric_names:
        if name in metrics:
            if mode == "median" and "median" in metrics[name]:
                value = metrics[name]["median"]
                print(f"{name.capitalize()}: {value:.{precision}f}")
            elif mode == "mean_std" and "mean" in metrics[name] and "std" in metrics[name]:
                mean = metrics[name]["mean"]
                std = metrics[name]["std"]
                print(f"{name.capitalize()}: {mean:.{precision}f} ± {std:.{precision}f}")
            else:
                print(f"{name.capitalize()}: Metric data incomplete")
        else:
            print(f"{name.capitalize()}: Metric not found")
    print("-" * 40)
