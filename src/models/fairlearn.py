from fairlearn.reductions import ExponentiatedGradient
# from fairlearn.postprocessing import ThresholdOptimizer


class ExpGradMitigator:
    def __init__(self, clf, constraint):
        self._clf = clf
        self._constraint = constraint
        self._mitigators = ExponentiatedGradient(clf, constraint)

    def fit_clf(self, X, y, **kwargs):
        self._clf.fit(X, y, **kwargs)

    def fit_mitigator(self, X, y, **kwargs):
        self._mitigators.fit(X, y, **kwargs)

    def mitigator_predict(self, X):
        return self._mitigators.predict(X)

    def clf_predict(self, X):
        return self._clf.predict(X)


# class ThresholdOptimizerMitigator:
#     def __init__(self, clf, constraint):
#         self._clf = clf
#         self._constraint = constraint
#         self._mitigators = ThresholdOptimizer(clf, constraint)

#     def fit_clf(self, X, y, **kwargs):
#         self._clf.fit(X, y, **kwargs)

#     def fit_mitigator(self, X, y, **kwargs):
#         self._mitigators.fit(X, y, **kwargs)

#     def clf_predict(self, X):
#         return self._clf.predict(X)

#     def mitigator_predict(self, X):
#         return self._mitigators.predict(X)
