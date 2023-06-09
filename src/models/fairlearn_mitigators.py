import numpy as np
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.postprocessing import ThresholdOptimizer


class ExpGradMitigator:
    """Exponentiated gradient mitigation algorithm."""
    def __init__(self, estimator_dict, constraints, seed=2023):
        self._estimator_dict = estimator_dict
        self._mitigator_dict = {}
        self._constraints = constraints
        self._seed = seed

    def get_estimator(self, mitigator_name):
        """Return the mitigator by name.

        Args:
            mitigator_name (str): Name of the classifier.

        Returns:
            sklearn classifier: The classifier.
        """
        return self._mitigator_dict[mitigator_name]

    def get_estimator_all(self):
        """Return all the mitigators.

        Returns:
            dict: The classifiers.
        """
        return self._mitigator_dict

    def fit_estimator(self, mitigator_name, X, y_true, sensitive_features):
        """Fit the mitigator by name and update mitigator dictionary.

        Args:
            mitigator_name (str): Name of the classifier.
            X (np.array): Features.
            y_true (np.array): Labels.
            sensitive_features (np.array): Sensitive features.
        """
        if self._seed:
            np.random.seed(self._seed)
        estimator = self._estimator_dict[mitigator_name]
        mitigator = ExponentiatedGradient(
            estimator=estimator,
            constraints=self._constraints
        )
        mitigator.fit(X, y_true, sensitive_features=sensitive_features)
        self._mitigator_dict[mitigator_name] = mitigator

    def fit_estimator_all(self, X, y_true, sensitive_features):
        """Fit all mitigators.

        Args:
            X (np.array): Features.
            y_true (np.array): Labels.
            sensitive_features (np.array): Sensitive features.
        """
        if self._seed:
            np.random.seed(self._seed)
        for mitigator_name in self._estimator_dict:
            self.fit_estimator(mitigator_name, X, y_true, sensitive_features)

    def predict(self, mitigator_name, X):
        """Predict the labels using the mitigator by name.

        Args:
            mitigator_name (str): Name of the classifier.
            X (np.array): Features.

        Returns:
            np.array: Predicted labels.
        """
        return self._mitigator_dict[mitigator_name].predict(X)

    def predict_all(self, X):
        """Predict the labels using all the mitigators.

        Args:
            X (np.array): Features.

        Returns:
            np.array: Predicted labels.
        """
        y_pred_dict = {}
        for mitigator_name in self._mitigator_dict:
            y_pred_dict[mitigator_name] = self.predict(mitigator_name, X)
        return y_pred_dict


def get_threshold_optim(X, y, sensitive, estimator_dict,
                        constraints='demographic_parity',
                        objective='accuracy_score'):
    """Get the threshold optimizer.

    Args:
        X (np.array): Features.
        y (np.array): Labels.
        sensitive (np.array): Sensitive features.
        estimator_dict (dict): Estimator dictionary.
        constraints (str): Fairness constraints.
        objective (str): Objective function.

    Returns:
        dict: Post-processing estimator dictionary.
    """
    postprocess_est = {}
    for mitigator_name in estimator_dict:
        postprocess_est[mitigator_name] = ThresholdOptimizer(
            estimator=estimator_dict[mitigator_name],
            constraints=constraints,
            objective=objective,
            predict_method='auto'
        )
        postprocess_est[mitigator_name].fit(X, y, sensitive_features=sensitive)
    return postprocess_est
