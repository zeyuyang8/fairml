from sklearn.tree import DecisionTreeClassifier


class SklearnClfs:
    """Container for sklearn classifiers."""
    def __init__(self):
        self._clf_dict = {
            "Decision tree classifier": DecisionTreeClassifier(max_depth=4),
        }

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
