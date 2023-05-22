from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count, selection_rate, mean_prediction
from fairlearn.metrics import false_negative_rate, false_positive_rate
from fairlearn.metrics import true_negative_rate, true_positive_rate


def customized_classification_report(y_true, y_pred, output_dict=True):
    """Customized classification report.

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels
        output_dict (bool): whether to output dictionary

    Returns:
        classification_report (dict): classification report
    """
    return classification_report(y_true, y_pred, output_dict=output_dict)


def eval_multi_clf_fairness(y_true, y_pred, sensitive_features):
    """Evaluate fairness of the multi-class classifier.

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels
        sensitive_features (pd.Series): sensitive features series

    Returns:
        metric_frame (fairlearn.metrics.MetricFrame): metric frame
    """
    metrics = {
        "accuracy": accuracy_score,
        "confusion matrix": confusion_matrix,
        "classication report": customized_classification_report
    }
    metric_frame = MetricFrame(metrics=metrics,
                               y_true=y_true,
                               y_pred=y_pred,
                               sensitive_features=sensitive_features)
    return metric_frame


def eval_binary_clf_fairness(y_true, y_pred, sensitive_features):
    """Evaluate fairness of the binary classifier.

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels
        sensitive_features (pd.Series): sensitive features series

    Returns:
        metric_frame (fairlearn.metrics.MetricFrame): metric frame
    """
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "confusion matrix": confusion_matrix,
        "classification report": customized_classification_report,
        "count": count,
        "mean prediction": mean_prediction,
        "selection rate": selection_rate,
        "false negative rate": false_negative_rate,
        "false positive rate": false_positive_rate,
        "true negative rate": true_negative_rate,
        "true positive rate": true_positive_rate
    }
    metric_frame = MetricFrame(metrics=metrics,
                               y_true=y_true,
                               y_pred=y_pred,
                               sensitive_features=sensitive_features)
    return metric_frame
