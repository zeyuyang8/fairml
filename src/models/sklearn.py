from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


class SklearnClfs:
    def __init__(self, clf_dict):
        self._clf_dict = clf_dict

    def add_estimator(self, clf_name, clf):
        self._clf_dict[clf_name] = clf

    def get_estimator(self, clf_name):
        return self._clf_dict[clf_name]

    def get_estimator_all(self):
        return self._clf_dict

    def fit_estimator(self, clf_name, X, y):
        self._clf_dict[clf_name].fit(X, y)

    def fit_estimator_all(self, X, y):
        for clf_name in self._clf_dict:
            self.fit_estimator(clf_name, X, y)

    def predict(self, clf_name, X):
        return self._clf_dict[clf_name].predict(X)

    def predict_all(self, X):
        return {clf_name: self.predict(clf_name, X) for clf_name in self._clf_dict}


DEFAULT_SKLEARN_CLFS = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Decistion tree classifier": DecisionTreeClassifier(max_depth=4),
    "Random forest classifier": RandomForestClassifier(max_depth=4),
    "AdaBoost classifier": AdaBoostClassifier(),
    "Multi-layer perceptron classifier": MLPClassifier(max_iter=1000)
}
