import numpy as np
from skquery import QueryStrategy


class Random(QueryStrategy):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, X, partition=None, oracle=None):
        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}
        candidates = [np.random.choice(range(X.shape[0]), size=2, replace=False).tolist() for _ in range(oracle.budget)]

        for i, j in candidates:
            must_linked = oracle.query(i, j)
            if must_linked:
                ml.append((i, j))
            else:
                cl.append((i, j))

        return constraints
