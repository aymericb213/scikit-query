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

    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.partition = []
        self.epsilon = epsilon
        self.fuzzy_partition = None
        self.p_dists = None

    def fit(self, X, oracle, **kwargs):
        X = self._check_dataset_type(X)

        if "partition" in kwargs:
            self.partition = kwargs["partition"]

        K = self._get_number_of_clusters(**kwargs)
        self.epsilon = self.epsilon * K

        self._pre_clustering(X, K)

        return self._marking(X, K, oracle)

    def _pre_clustering(self, X, k):
        cntr, u, u0, d, jm, p, fpc = fuzzy.cmeans(X.T, k, 2, 0.05, maxiter=1000)
        self.fuzzy_partition = u
        self.p_dists = d

    def _marking(self, X, k, oracle):
        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}

        entropies = np.array([entropy([self.fuzzy_partition[k][i] for k in range(k)]) for i in range(X.shape[0])])  # récupérer les weak samples (voir discord)
        weak_samples = np.where(entropies > np.log(k) - self.epsilon)[0]
        sorted_weak = reversed(sorted(weak_samples, key=lambda x: entropies[x]))

        strong_samples = self._compute_medoids()  # récupération des strong samples.

        while True:
            try:
                if len(weak_samples) == 0:
                    raise QueryNotFoundError

                try:
                    xweak = next(sorted_weak)
                except StopIteration:
                    raise QueryNotFoundError
                strong_samples.sort(key=lambda x: self._symmetric_relative_entropy(xweak, x, k))
                first_strong = strong_samples[0][1]
                second_strong = strong_samples[1][1]

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
        u_xj = [self.fuzzy_partition[k][x[1]] for k in range(K)]  # x[1] correspond à l'indice dans la data
        u_xweak = [self.fuzzy_partition[k][xweak] for k in range(K)]
        return (entropy(u_xj, u_xweak) + entropy(u_xweak, u_xj)) / 2

    def _compute_medoids(self):
        medoids = []  # liste de tuple (indice du centre, indice du medoid)
        for i in range(len(self.p_dists)):
            dist_min = self.p_dists[i][0]
            minInd = 0
            for ind, e in enumerate(self.p_dists[i]):
                if e < dist_min:
                    dist_min = e
                    minInd = ind
            medoids.append((i, minInd))

        return medoids
