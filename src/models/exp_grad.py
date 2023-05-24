import numpy as np
from fairlearn.reductions import ExponentiatedGradient


def exp_grad_est(est, constraint, X, y_true, sensitive, seed=2023):
    if seed:
        np.random.seed(seed)
    mitigator = ExponentiatedGradient(
        estimator=est,
        constraints=constraint
    )
    mitigator.fit(X, y_true, sensitive_features=sensitive)
    return mitigator