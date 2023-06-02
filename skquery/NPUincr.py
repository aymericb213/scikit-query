import numpy as np
from skquery.oracle.MLCLOracle import MaximumQueriesExceeded
from skquery import QueryStrategy
from sklearn.ensemble import RandomForestClassifier


class NPUincr(QueryStrategy):
    """ Incremental version of NPU. Based off the implementation at
    https://github.com/datamole-ai/active-semi-supervised-clustering
    """

    def __init__(self, neighborhoods=None):
        super().__init__()
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods

    def fit(self, X, partition, oracle):
        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}
        self.partition = partition
        n = X.shape[0]

        if not self.neighborhoods:
            # Initialization
            x_i = np.random.choice(list(range(n)))
            self.neighborhoods.append([x_i])

        while True:
            try:
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

            except MaximumQueriesExceeded:
                break

        return constraints

    def _most_informative(self, X, n_trees=50):
        n = X.shape[0]
        nb_neighborhoods = len(self.neighborhoods)

        neighborhoods_union = set()
        for neighborhood in self.neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        unqueried_indices = set(range(n)) - neighborhoods_union

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
                uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i)).sum()
                expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
            else:
                uncertainties[x_i] = 0
                expected_costs[x_i] = 1  # ?

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]
