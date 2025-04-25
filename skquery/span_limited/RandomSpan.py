import numpy as np
from ..strategy import QueryStrategy


class RandomSpan(QueryStrategy):
    def __init__(self, size=10, generic=False):
        super().__init__()
        self.size = size
        self.generic = generic

    def fit(self, X, oracle, partition=None):
        np.random.seed(9)
        constraints = {"span": []}
        candidates = [np.random.choice(range(X.shape[0]), size=self.size, replace=False).tolist() for _ in range(oracle.budget)]

        for group in candidates:
            clusters = set()
            for x in group:
                clusters.add(oracle.truth[x])

            if self.generic:
                constraints["span"].append((group, len(clusters)))
            else:
                constraints["span"].append((group, clusters))
        return constraints
