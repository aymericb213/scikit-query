import numpy as np
from ..strategy import QueryStrategy


class RandomMLCL(QueryStrategy):
    def __init__(self):
        super().__init__()

    def fit(self, X, oracle, **kwargs):
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
