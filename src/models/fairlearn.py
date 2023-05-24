import numpy as np
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.postprocessing import ThresholdOptimizer


# Exponentiated gradient
def exp_grad_est(est, constraint, X, y_true, sensitive, seed=2023):
    if seed:
        np.random.seed(seed)
    mitigator = ExponentiatedGradient(
        estimator=est,
        constraints=constraint
    )
    mitigator.fit(X, y_true, sensitive_features=sensitive)
    return mitigator


# Grid search
def grid_search(est, constraint, grid_size, X, y_true, sensitive):
    sweep = GridSearch(
        est,
        constraint,
        grid_size
    )
    sweep.fit(X, y_true, sensitive_features=sensitive)


# Threshold optimizer
def threshold_opt(est, constraint, objective, X, y_true, sensitive):
    mitigator = ThresholdOptimizer(
        estimator=est,
        constraints=constraint,
        objective=objective,
        predict_method='auto'
    )
    mitigator.fit(X, y_true, sensitive_features=sensitive)
    return mitigator

# Correlation remover

# Adversarial fairness classifier

# Adversarial fairness regressor
