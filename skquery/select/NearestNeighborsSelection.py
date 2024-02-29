import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..strategy import QueryStrategy


class NearestNeighborsSelection(QueryStrategy):

    def __init__(self, m, distances=None):
        """
        Selects the most informative samples according to the Cai et al. strategy,
        which selects the samples that have neighbors from different clusters.

        Parameters
        ----------
        m: int or float
            Number of nearest neighbors to use for the selection.
            If a float is provided, it will be interpreted as a percentage of the dataset size.
        distances: array-like, default=None
            Precomputed distances between samples. If None, the distances will be computed on the fly.
        """
        super().__init__()
        if 0 < m < 1:
            if distances is not None:
                self.n_neighbors = int(np.round(m * distances.shape[0]))
            else:
                self.n_neighbors = m
        elif m >= 1:
            self.n_neighbors = int(m)
        else:
            raise ValueError("m must be a positive number.")
        self.p_dist = [] if distances is None or distances.shape == 0 else distances

    def select(self, X, partition, return_best=False):
        """
        Selects the most informative samples according to the Cai et al. strategy.

        Parameters
        ----------
        X: array-like
            Dataset to select from.
        partition: array-like
            Partition of the dataset.
        return_best: bool, default=False
            Whether to return the best sample along with the set of selected samples.

        Returns
        -------
        selected: list
            List of selected samples.
        best: int
            Index of the best sample, if ``return_best`` is True.
        """
        X = self._check_dataset_type(X)
        if 0 < self.n_neighbors < 1:
            self.n_neighbors = int(np.round(self.n_neighbors * X.shape[0]))

        # Compute the nearest neighbors
        if len(self.p_dist) == 0:
            nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            nn.fit(X)
        else:
            nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="precomputed")
            nn.fit(self.p_dist)
        nn_dists, nn_idx = nn.kneighbors()

        # Select the samples that have neighbors from different clusters
        selected = []
        q = (0, 0)
        for i in range(X.shape[0]):
            n_diffs = 0
            for n in nn_idx[i]:
                if partition[i] != partition[n]:
                    n_diffs += 1
            if n_diffs > 0:
                selected.append(i)
                if n_diffs > q[1]:
                    q = (i, n_diffs)
        return selected, q[0] if return_best else selected
