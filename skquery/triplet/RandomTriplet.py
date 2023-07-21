import numpy as np
from ..strategy import QueryStrategy
from ..exceptions import NoAnswerError


class RandomTriplet(QueryStrategy):
    """
    Random sampling of triplet constraints.
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """
        Selects triplet constraints randomly.

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

        triplet = []
        constraints = {"triplet": triplet}
        candidates = [np.random.choice(range(X.shape[0]), size=3, replace=False).tolist() for _ in range(oracle.budget)]

        for i, j, k in candidates:
            try:
                if oracle.query(i, j, k):
                    triplet.append((i, j, k))
                else:
                    triplet.append((j, k, i))
            except NoAnswerError:
                continue

        return constraints
