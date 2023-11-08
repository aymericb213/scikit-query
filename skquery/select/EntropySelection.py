import numpy as np
import skfuzzy as fuzzy
from scipy.stats import entropy
from ..strategy import QueryStrategy


class EntropySelection(QueryStrategy):
    """
    Selects the most informative samples according to the entropy of their soft partition.
    Strategy based on Chen and Jin (2020).
    """

    def __init__(self, m):
        super().__init__()
        self.percent = m

    def select(self, X, partition, return_best=False, fuzziness=2, tolerance=10**-5, max_iter=100):
        """
        Selects the most informative samples according to the entropy of their soft partition.

        Parameters
        ----------
        X: array-like
            Dataset to select from.
        partition: array-like
            Partition of the dataset.
        return_best: bool, default=False
            Whether to return the best sample along with the set of selected samples.
        fuzziness: float, default=2
            Fuzziness parameter for the fuzzy c-means algorithm.
        tolerance: float, default=10**-5
            Tolerance parameter for the fuzzy c-means algorithm.
        max_iter: int, default=100
            Maximum number of iterations for the fuzzy c-means algorithm.

        Returns
        -------
        selected: list
            List of selected samples.
        best: int
            Index of the best sample, if ``return_best`` is True.
        """
        cntr, u, u0, d, jm, p, fpc = fuzzy.cmeans(X.T, len(set(partition)), fuzziness, tolerance, maxiter=max_iter)

        if 0 < self.percent < 1:
            coeff = self.percent * X.shape[0]
        elif 1 < self.percent <= X.shape[0]:
            coeff = self.percent
        else:
            raise ValueError("m must be a positive number.")
        entropies = np.array([entropy([u[c][i] for c in range(len(set(partition)))]) for i in range(X.shape[0])])
        desc_entropies = np.flip(np.sort(entropies))
        threshhold = desc_entropies[int(np.round(coeff))]
        selected = [i for i in range(X.shape[0]) if entropies[i] >= threshhold]
        return selected, np.argmax(entropies) if return_best else selected
