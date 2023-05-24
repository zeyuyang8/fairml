import pandas as pd


def num_to_percentile_category(target, num_classes=4):
    y_true_categorical = pd.qcut(target, num_classes, labels=range(num_classes))
    return y_true_categorical
