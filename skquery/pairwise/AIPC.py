"""
 Active Informative Pairwise Constraint Formulation algorithm from Zhong et al. 2019.
"""
# Authors : Brice Jacquesson, Matthéo Pailler, Aymeric Beauchamp

from ..exceptions import EmptyBudgetError, QueryNotFoundError, NoAnswerError
from ..strategy import QueryStrategy
import numpy as np
from scipy.stats import entropy
import skfuzzy as fuzzy


class AIPC(QueryStrategy):
    """
    Active Informative Pairwise Constraint Formulation algorithm [1]_.

    Selects pairwise constraints between strong and weak samples,
    as defined according to Shannon entropy based on fuzzy clustering.

    Parameters
    ----------
    epsilon : float, default=0.
        Threshold value for the entropy of weak samples.
        By default, ``epsilon`` is computed during fit.

    Attributes
    ----------
    epsilon : float
        Threshold value for the entropy of weak samples.
        The value can be given at initialization,
        otherwise it is computed during fit.
    fuzzy_partition : array-like
        Soft partition of the dataset computed by fuzzy c-means clustering.

    References
    ----------
    .. [1] Zhong, G., Deng, X., Shengbing, X. Active Informative
           Pairwise Constraint Formulation Algorithm for Constraint-Based
           Clustering. 2019. IEEE Access Volume 7.
    """
    def __init__(self, epsilon=0.):
        super().__init__()
        self.epsilon = epsilon
        self.fuzzy_partition = None

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """
        Select pairwise constraints with AIPC.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : array-like
            Existing partition of the data.
            Used to define the number of clusters.
        n_clusters : int
            Number of clusters to find.

        Returns
        -------
        constraints : dict of lists
            ML and CL constraints between strong and weak samples.
        """
        X = self._check_dataset_type(X)
        K = self._get_number_of_clusters(partition, n_clusters)

        p_dists = self._pre_clustering(X, K)

        entropies = np.array([entropy([self.fuzzy_partition[k][i] for k in range(K)]) for i in range(X.shape[0])])

        assert self.epsilon >= 0.
        if self.epsilon == 0.:
            self._epsilon(entropies, K, oracle.budget)

        return self._marking(entropies, K, p_dists, oracle)

    def _epsilon(self, entropies, k, n_queries):
        """
        Computes the threshold value of Shannon entropy
        over which a sample is considered weak.

        Parameters
        ----------
        entropies : array-like
            Shannon entropy of every point in the dataset.
        k : int
            Number of clusters used for fuzzy clustering.
        n_queries : int
            Number of queries to ask, i.e. the oracle's budget.
        """
        desc_entropies = np.flip(np.sort(entropies))
        self.epsilon = np.log(k) - desc_entropies[n_queries]

    def _pre_clustering(self, X, k, fuzziness=2, tolerance=10**-5, max_iter=100):
        """
        Computes a soft partition of data with
        fuzzy c-means clustering.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        k : int
            Number of clusters used for fuzzy clustering.
        fuzziness : int, default=2
            Degree of fuzziness.
        tolerance : int, default=10e-5
            Stopping criterion : minimal difference between two iterations of cmeans.
        max_iter : int, default=100
            Maximum number of c-means iterations.

        Notes
        -----
        Default values for ``fuzziness``, ``tolerance`` and ``max_iter``
        replicate the experimental setup in the article.
        """
        cntr, u, u0, d, jm, p, fpc = fuzzy.cmeans(X.T, k, fuzziness, tolerance, maxiter=max_iter)
        self.fuzzy_partition = u
        return d

    def _marking(self, entropies, k, p_dists, oracle):
        """
        Selects pairwise constraints based on
        symmetric relative entropy between strong and
        weak samples.

        Params
        ------
        entropies : array-like
            Shannon entropy of every point in the dataset.
        k : int
            Number of clusters used for fuzzy clustering.
        p_dists : array-like
            Euclidean distance matrix between instances and fuzzy centers.
        oracle : callable
            Source of background knowledge able to answer the queries.

        Returns
        -------
        constraints : dict of lists
            Selected pairwise constraints.
        """
        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}

        weak_samples = np.where(entropies > np.log(k) - self.epsilon)[0]
        sorted_weak = reversed(sorted(weak_samples, key=lambda x: entropies[x]))

        # Medoids of fuzzy cluster centers
        strong_samples = [(i, np.argmin(p_dists[i])) for i in range(k)]

        while True:
            try:
                if len(weak_samples) == 0:
                    raise QueryNotFoundError

                try:
                    xweak = next(sorted_weak)
                except StopIteration:
                    raise QueryNotFoundError
                strong_samples.sort(key=lambda x: self._symmetric_relative_entropy(xweak, x, k))
                first_strong = strong_samples[0][0]
                second_strong = strong_samples[1][0]

                try:
                    if oracle.query(xweak, second_strong):
                        ml.append((xweak, second_strong))
                    else:
                        cl.append((xweak, second_strong))
                        ml.append((xweak, first_strong))
                except NoAnswerError:
                    continue

            except (EmptyBudgetError, QueryNotFoundError):
                break
        return constraints

    def _symmetric_relative_entropy(self, xweak, x, K):
        """
        Computes symmetric relative entropy between a weak sample and a strong one.

        Parameters
        ----------
        xweak : int
            Index of weak sample.
        x : int
            Index of strong sample.
        K : int
            Number of clusters in fuzzy clustering.

        Returns
        -------
        s_entropy : float
            Symmetric relative entropy between the weak and strong samples.
        """
        u_xj = [self.fuzzy_partition[k][x[1]] for k in range(K)]  # x[1] correspond à l'indice dans la data
        u_xweak = [self.fuzzy_partition[k][xweak] for k in range(K)]
        return (entropy(u_xj, u_xweak) + entropy(u_xweak, u_xj)) / 2
