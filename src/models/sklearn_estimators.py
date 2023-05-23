from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


class SklearnClfs:
    """Container for sklearn classifiers."""
    def __init__(self, clf_dict):
        """Initialize the container.

        Args:
            clf_dict (dict): Dictionary of classifiers.

        Example:
            from sklearn.tree import DecisionTreeClassifier
            clf_dict = {
                "Decision tree clf": DecisionTreeClassifier(max_depth=4),
            }
        """
        self._clf_dict = clf_dict

    def add_clf(self, clf_name, clf):
        """Add a new classifier.

        Args:
            clf_name (str): Name of the classifier.
            clf (sklearn classifier): The classifier.
        """
        self._clf_dict[clf_name] = clf

    def get_clf(self, clf_name):
        """Return the classifier by name.

        Args:
            clf_name (str): Name of the classifier.

        Returns:
            sklearn classifier: The classifier.
        """
        return self._clf_dict[clf_name]

    def get_clf_all(self):
        """Return all the classifiers.

        Returns:
            dict: The classifiers.
        """
        return self._clf_dict

    def fit_clf(self, clf_name, X, y):
        """Fit the classifier by name.

        Args:
            clf_name (str): Name of the classifier.
            X (np.array): Features.
            y (np.array): Labels.
        """
        self._clf_dict[clf_name].fit(X, y)

    def fit_clf_all(self, X, y):
        """Fit all the classifiers.

        Args:
            X (np.array): Features.
            y (np.array): Labels.
        """
        for clf_name in self._clf_dict:
            self.fit_clf(clf_name, X, y)

    def predict(self, clf_name, X):
        """Predict the labels using the classifier by name.

        Args:
            clf_name (str): Name of the classifier.
            X (np.array): Features.

        Returns:
            np.array: Predicted labels.
        """
        return self._clf_dict[clf_name].predict(X)

    def predict_all(self, X):
        """Predict the labels using all the classifiers.

        Args:
            X (np.array): Features.

        Returns:
            dict: Predicted labels.
        """
        return {clf_name: self.predict(clf_name, X) for clf_name in self._clf_dict}


DEFAULT_SKLEARN_CLFS = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Decistion tree classifier": DecisionTreeClassifier(max_depth=4),
    "Random forest classifier": RandomForestClassifier(max_depth=4),
    "AdaBoost classifier": AdaBoostClassifier(),
    "Multi-layer perceptron classifier": MLPClassifier(max_iter=1000)
}
