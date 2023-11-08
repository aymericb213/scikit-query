import numpy as np
from ..strategy import QueryStrategy


class RandomSelection(QueryStrategy):
    """
    Random selection of points.
    """
    def __init__(self, m):
        super().__init__()
        self.percent = m

    def select(self, X, partition=None, return_best=False):
        """
        Selects a random subset of the dataset.

        Parameters
        ----------
        X: array-like
            Dataset to select from.
        partition: Ignored
            Not used, present for API consistency.

        Returns
        -------
        selected: list
            List of selected samples.
        """
        X = self._check_dataset_type(X)
        size = 0
        if 0 < self.percent < 1:
            size = int(np.round(self.percent * X.shape[0]))
        elif 1 < self.percent <= X.shape[0]:
            size = self.percent
        selected = np.random.choice(X.shape[0], size=size, replace=False).tolist()
        return selected, selected[np.random.randint(0, len(selected))] if return_best else selected
