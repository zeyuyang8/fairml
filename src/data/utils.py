import pandas as pd


def num_to_percentile_category(target, num_classes=4):
    """Convert numerical target to categorical target based on quantile.

    Args:
        target (pd.Series): target series
        num_classes (int): number of classes

    Returns:
        target (pd.Series): categorical target series
    """
    y_true_categorical = pd.qcut(target, num_classes, labels=range(num_classes))
    return y_true_categorical
