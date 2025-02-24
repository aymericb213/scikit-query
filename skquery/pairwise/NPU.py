""" Normalized Point-wise Uncertainty algorithm. Based off the implementation at
https://github.com/datamole-ai/active-semi-supervised-clustering
"""
# Authors : Aymeric Beauchamp

import numpy as np
from ..exceptions import EmptyBudgetError, QueryNotFoundError, NoAnswerError
from ..strategy import QueryStrategy
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy, rv_discrete


class NPU(QueryStrategy):
    """
    Normalized Point-based Uncertainty algorithm [1]_.
    
    Uncertainty sampling using Shannon entropy
    based on an existing partition, normalized by
    the expected number of queries for a given point.
    
    Parameters
    ----------
    neighborhoods : list of lists, default=None
        Optional precomputed neighborhoods of the data.
        This can be used as a warm start, e.g. for incremental constrained clustering.
    cc_alg : callable, default=None
        Constrained clustering algorithm to use in tandem with
        the selection procedure.
        Not needed if a partition is given during fit.
        
    Attributes
    ----------
    partition : array-like
        Partition of the dataset.
    neighborhoods : list of lists
        Points whose cluster affectation is certified by the answers of the oracle to the queries.
        Each list contains points belonging to the same cluster.
    cc_alg : callable
        Constrained clustering algorithm used to compute a partition
        using the selected constraints.
    unknown : set
        Points whose affectation has been queried but remain unknown, i.e. the oracle
        couldn't answer the query.
    
    Notes
    -----
    This implementation of NPU works either with a constrained clustering algorithm or
    by giving a partition as argument of the ``fit`` method. In the former case,
    only algorithms from ``active-semi-supervised-clustering`` library are supported.

    References
    ----------
    .. [1] Xiong, S., Azimi, J., Fern, X. Z. Active Learning of Constraints for
           Semi-Supervised Clustering. 2013. IEEE Transactions on Knowledge and
           Data Engineering Volume 26, 1, pp 43-54.
    """
    def __init__(self, neighborhoods=None, cc_alg=None):
        super().__init__()
        self.partition = []
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods
        self.cc_alg = cc_alg
        self.unknown = set()

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """Select pairwise constraints with NPU.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : array-like
            Existing partition of the data.
            Not used if a clustering algorithm has been defined at init.
        n_clusters : Ignored
            Not used, present for API consistency.

        Returns
        -------
        constraints : dict of lists
            ML and CL constraints derived from the neighborhoods.
        """
        X = self._check_dataset_type(X)

        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}
        if not self.cc_alg:
            if partition is not None:
                self.partition = partition
            else:
                raise AttributeError("NPU requires either a CC algorithm or a partition.")

        if len(self.neighborhoods) == 0:
            # Initialization
            x_i = np.random.choice(list(range(X.shape[0])))
            self.neighborhoods.append([x_i])

        while True:
            try:
                if self.cc_alg is not None:
                    self.cc_alg.fit(X.to_numpy(), ml=ml, cl=cl)
                    self.partition = self.cc_alg.labels_

                x_i, p_i = self._most_informative(X)

                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, self.neighborhoods)))))[1]

                must_link_found = False

                # The oracle determines the neighborhood of x_i
                try:
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
                except NoAnswerError:
                    self.unknown.add(x_i)

            except (EmptyBudgetError, QueryNotFoundError):
                break

        return constraints

    def _most_informative(self, X, n_trees=50):
        """
        Selects the most informative instance in the dataset
        by learning a random forest from the partition and
        computing entropy based on leaf pairings.
        
        Parameters
        ----------
        X : array-like
            Instances to use for query.
        n_trees : int, default=50
            Number of estimators in the random forest. Default is 50 as per the original paper.
        
        Returns
        -------
        x_i, p_i : int, array-like
            Index of the most informative point and its probability distribution.
        """
        n = X.shape[0]
        nb_neighborhoods = len(self.neighborhoods)

        neighborhoods_union = set([x for nbhd in self.neighborhoods for x in nbhd])
        unqueried_indices = set(range(n)) - neighborhoods_union - self.unknown

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
                expected_costs[x_i] = rv_discrete(values=(range(1, len(positive_p_i) + 1), np.flip(sorted(positive_p_i)))).expect()
            else:
                # case where neighborhood affectation is certain
                uncertainties[x_i] = 0
                expected_costs[x_i] = 1

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]
