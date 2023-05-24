import matplotlib.pyplot as plt
import numpy as np


def plot_val_counts(features, col_name, xlabel, ylabel, title,
                    xmapping, save_path=None):
    """Plot bar chart visualizing the value counts of
    a column in the dataframe.

    Args:
        features (pd.Dataframe): features dataframe
        col_name (str): column name
        xmapping (dict): mapping from the value to the label
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of the plot
        save_path (str): path to save the plot

    Returns:
        fig (matplotlib.figure.Figure): figure
    """
    fig, ax = plt.subplots()
    val_counts_series = features[col_name].value_counts()
    if xmapping:
        val_counts_series = val_counts_series.rename(xmapping)
    keys = list(val_counts_series.keys())
    vals = list(val_counts_series.values)
    ax.bar(keys, vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(keys)
    ax.set_xticklabels(keys, rotation=45, ha='right')
    if save_path:
        fig.savefig(save_path + title + ".png", bbox_inches='tight')
    return fig


def plot_corr_heatmap(features, targets, title, save_path=None):
    """Plot correlation heatmap for the dataframe.

    Args:
        features (pd.Dataframe): features dataframe
        targets (pd.Series): target labels
        title (str): title of the plot
        save_path (str): path to save the plot

    Returns:
        fig (matplotlib.figure.Figure): figure
    """
    data = features.copy()
    data['target'] = targets
    cols = list(data.columns)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(round(data.corr(), 2), cmap="coolwarm")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(cols)), labels=cols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                round(data.corr().to_numpy()[i, j], 2),
                ha="center",
                va="center",
            )
    ax.set_title(f"{title}")

    if save_path:
        fig.savefig(save_path + title + ".png", bbox_inches='tight')
    return fig


def plot_target_dist(target, xlabel, ylabel, title, save_path=None):
    """Plot target distribution in a bar chart.

    Args:
        target (pd.Series): series
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): title of the plot
    """
    print(target.describe())
    fig, ax = plt.subplots()
    ax.hist(target, bins=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path + title + ".png", bbox_inches='tight')
    return fig


def plot_binary_fairness_metrics_by_group(metric_frame, xticks, title,
                                          keys, save_path=None):
    """Plot binary fairness metrics by group.

    Args:
        metric_frame (fairlearn.metrics.MetricFrame): metric frame
        xticks (list): x-axis ticks
        title (str): title of the plot
        keys (list): keys to plot
        save_path (str): path to save the plot
    """
    plt.figure(figsize=(18, 18))
    reindexed_metric_frame = metric_frame.by_group.copy()
    reindexed_metric_frame.index = xticks
    reindexed_metric_frame.index.name = metric_frame.by_group.index.name
    reindexed_metric_frame[keys].plot(
        kind="bar",
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        rot=45,
        position=0,
        title=title
    )
    if save_path:
        plt.savefig(save_path + title + ".png", bbox_inches='tight')
