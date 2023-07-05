import numpy as np
from ..strategy import QueryStrategy


class RandomMLCL(QueryStrategy):
    """
    Random sampling of pairwise constraints.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """
        Selects pairwise constraints randomly.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : Ignored
            Not used, present for API consistency.
        n_clusters : Ignored
            Not used, present for API consistency.

        """
        X = self._check_dataset_type(X)

        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}
        candidates = [np.random.choice(range(X.shape[0]), size=2, replace=False).tolist() for _ in range(oracle.budget)]

        for i, j in candidates:
            if oracle.query(i, j):
                ml.append((i, j))
            else:
                cl.append((i, j))

        return constraints
