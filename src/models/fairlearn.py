class FairLearnMitigator:
    """Fair learn mitigator container."""
    def __init__(self):
        """Initialize."""
        self._clf = None
        self._mitigator = None

    def exponentiated_gradient(clf, constraint):
        """Exponentiated gradient algorithm.

        Args:
            clf: a classifier
            constraint: a constraint

        Returns:
            a mitigator
        """
        pass

    def grid_search(clf, constraint, grid_size=10):
        """Grid search algorithm.

        Args:
            clf: a classifier
            constraint: a constraint
            grid_size: grid size

        Returns:
            a mitigator
        """
        pass

    def threshold_optimizer(clf, constraint):
        """Threshold optimizer algorithm.

        Args:
            clf: a classifier
            constraint: a constraint

        Returns:
            a mitigator
        """
        pass
