""" Normalized Point-wise Uncertainty algorithm. Based off the implementation at
https://github.com/datamole-ai/active-semi-supervised-clustering
"""
# Authors : Aymeric Beauchamp

import numpy as np
from ..exceptions import EmptyBudgetError, QueryNotFoundError
from ..strategy import QueryStrategy
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy, rv_discrete


class NPU(QueryStrategy):

    def __init__(self, neighborhoods=None, clusterer=None):
        super().__init__()
        self.partition = []
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods
        self.clusterer = clusterer

    def fit(self, X, oracle, **kwargs):
        X = self._check_dataset_type(X)

        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}
        if not self.clusterer and "partition" in kwargs:
            # disregard if a CC algorithm is provided
            self.partition = kwargs["partition"]

        if len(self.neighborhoods) == 0:
            # Initialization
            x_i = np.random.choice(list(range(X.shape[0])))
            self.neighborhoods.append([x_i])

        while True:
            try:
                if self.clusterer is not None:
                    # only works with CC algorithms from active-semi-supervised-clustering library
                    self.clusterer.fit(X.to_numpy(), ml=ml, cl=cl)
                    self.partition = self.clusterer.labels_

                x_i, p_i = self._most_informative(X)

                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, self.neighborhoods)))))[1]
                #  print(x_i, self.neighborhoods, p_i, sorted_neighborhoods)

                must_link_found = False

                # The oracle determines the neighborhood of x_i
                for neighborhood in sorted_neighborhoods:

                    must_linked = oracle.query(x_i, neighborhood[0])
                    if must_linked:
                        for x_j in neighborhood:
                            ml.append((x_i, x_j))

                        for other_neighborhood in self.neighborhoods:
                            if neighborhood != other_neighborhood:
                                for x_j in other_neighborhood:
                                    cl.append((x_i, x_j))

                        neighborhood.append(x_i)
                        must_link_found = True
                        break
                        # TODO should we add the cannot-link in case the algorithm stops before it queries all neighborhoods?

                if not must_link_found:
                    for neighborhood in self.neighborhoods:
                        for x_j in neighborhood:
                            cl.append((x_i, x_j))

                    self.neighborhoods.append([x_i])

            except (EmptyBudgetError, QueryNotFoundError):
                break

        return constraints

    def _most_informative(self, X, n_trees=50):
        n = X.shape[0]
        nb_neighborhoods = len(self.neighborhoods)

        neighborhoods_union = set([x for nbhd in self.neighborhoods for x in nbhd])
        unqueried_indices = set(range(n)) - neighborhoods_union

        if unqueried_indices == set():
            raise QueryNotFoundError
        if nb_neighborhoods <= 1:
            return np.random.choice(list(unqueried_indices)), [1]

        # Learn a random forest classifier
        rf = RandomForestClassifier(n_estimators=n_trees)
        rf.fit(X, self.partition)

        # Compute the similarity matrix
        leaf_indices = rf.apply(X)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
        S = S / n_trees

        p = np.empty((n, nb_neighborhoods))
        uncertainties = np.zeros(n)
        expected_costs = np.ones(n)

        # For each point that is not in any neighborhood...
        for x_i in unqueried_indices:
            for n_i in range(nb_neighborhoods):
                p[x_i, n_i] = (S[x_i, self.neighborhoods[n_i]].sum() / len(self.neighborhoods[n_i]))

            # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
            if np.all(p[x_i, ] == 0):
                p[x_i, ] = np.ones(nb_neighborhoods)

            p[x_i, ] = p[x_i, ] / p[x_i, ].sum()

            if not np.any(p[x_i, ] == 1):
                positive_p_i = p[x_i, p[x_i, ] > 0]
                uncertainties[x_i] = entropy(positive_p_i, base=2)
                expected_costs[x_i] = rv_discrete(values=(range(1, len(positive_p_i) + 1), positive_p_i)).expect()
            else:
                # case where neighborhood affectation is certain
                uncertainties[x_i] = 0
                expected_costs[x_i] = 1

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]
