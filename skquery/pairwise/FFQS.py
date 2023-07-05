import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ..exceptions import EmptyBudgetError, QueryNotFoundError, NoAnswerError
from ..strategy import QueryStrategy


class FFQS(QueryStrategy):
    """
    Farthest First Query Selection [1]_.

    Explores the data with farthest-first traversal to discover the cluster structure,
    then consolidates the skeleton with random selection.

    Parameters
    ----------
    neighborhoods : list of lists, default=None
        Optional precomputed neighborhoods of the data.
        This can be used as a warm start, e.g. to skip the Explore phase.
    distances : array-like, default=None
        Euclidean pairwise distance matrix of the data.
        If none given, it will be computed during fit.

    Attributes
    ----------
    neighborhoods : list of lists
        Points whose cluster affectation is certified by the answers of the oracle to the queries.
        Each list contains points belonging to the same cluster.
    p_dists : array-like
        Euclidean pairwise distance matrix of the data.
    unknown : set
        Points whose affectation has been queried but remain unknown, i.e. the oracle
        couldn't answer the query.

    References
    ----------
    .. [1] Basu, S., Banerjee, A., Mooney, R. Active Semi-Supervision for
           Pairwise Constrained Clustering. 2004. Proceedings of the 2004 SIAM International
           Conference on Data Mining. ISBN 978-1-61197-274-0, pp. 333-344.
    """
    def __init__(self, neighborhoods=None, distances=None):
        super().__init__()
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods
        self.p_dists = [] if distances is None or distances.shape == 0 else distances
        self.unknown = set()

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """Select pairwise constraints with the FFQS scheme.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : array-like, default=None
            Existing partition of the data that provides the number of clusters to find.
            It has priority over ``n_clusters``, i.e. ``n_clusters`` will not be
            taken into account if a partition is passed.
        n_clusters : int, default=None
            Number of clusters to find.
            If none given, only the Explore phase will be used.

        Returns
        -------
        constraints : dict of lists
            ML and CL constraints derived from the neighborhoods.
        """
        X = self._check_dataset_type(X)
        K = self._get_number_of_clusters(partition, n_clusters)

        if len(self.p_dists) == 0:
            self.p_dists = pd.DataFrame(squareform(pdist(X)))

        if K == 0 or 0 <= len(self.neighborhoods) < K:
            self._explore(X, K, oracle)
        if oracle.queries < oracle.budget:
            self._consolidate(X, oracle)

        ml, cl = self.get_constraints_from_neighborhoods()
        constraints = {"ml": ml, "cl": cl}
        return constraints

    def _explore(self, X, k, oracle):
        """
        Explore phase of FFQS.
        Farthest-first traversal until k clusters are found or
        the budget is depleted.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        k : int
            Number of clusters to find.
        oracle : callable
            Source of background knowledge able to answer the queries.
        """
        traversed = []
        if not self.neighborhoods:
            # Initialization
            x = np.random.choice(X.shape[0])
            self.neighborhoods.append([x])
            traversed.append(x)
        else:
            # Retrieving past traversed points from neighborhoods
            traversed = [x for nbhd in self.neighborhoods for x in nbhd]

        unqueried_indices = list(set(range(X.shape[0])) - set(traversed) - self.unknown)

        # Use all the budget if k is not known
        while k == 0 or len(self.neighborhoods) < k:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                # Farthest point : max distance to the set of traversed points
                # d(point, set) = min([d(point, x) for x in set])
                farthest = self.p_dists.iloc[unqueried_indices, traversed].min(axis=1).idxmax()

                # Querying neighborhood of farthest point
                try:
                    new_neighborhood = True
                    for neighborhood in self.neighborhoods:
                        if oracle.query(farthest, neighborhood[0]):
                            neighborhood.append(farthest)
                            new_neighborhood = False
                            break
                    if new_neighborhood:
                        self.neighborhoods.append([farthest])
                    traversed.append(farthest)
                except NoAnswerError:
                    self.unknown.add(farthest)

                unqueried_indices.remove(farthest)

            except (EmptyBudgetError, QueryNotFoundError):
                break

    def _consolidate(self, X, oracle):
        """
        Consolidate phase of FFQS.
        Pick points at random and query them against the k neighborhoods
        found in Explore.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        """
        neighborhoods_union = set([x for nbhd in self.neighborhoods for x in nbhd])

        unqueried_indices = set(range(X.shape[0])) - neighborhoods_union - self.unknown

        while True:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                i = np.random.choice(list(unqueried_indices))

                sorted_neighborhoods = sorted(self.neighborhoods, key=lambda nbhd: self.p_dists.iloc[i, nbhd].min())

                try:
                    for neighborhood in sorted_neighborhoods:
                        if oracle.query(i, neighborhood[0]):
                            neighborhood.append(i)
                            break
                    neighborhoods_union.add(i)
                except NoAnswerError:
                    self.unknown.add(i)

                unqueried_indices.remove(i)

            except (EmptyBudgetError, QueryNotFoundError):
                break

    def get_constraints_from_neighborhoods(self):
        """
        Derives constraints from the neighborhood structure :
        ML between all pairs of points in the same neighborhood and
        CL between all pairs of points in separate neighborhoods

        Returns
        -------
        ml, cl : list of tuples
            Pairwise constraints derived from the neighborhoods.
        """
        ml, cl = [], []

        for neighborhood in self.neighborhoods:

            for i in neighborhood:
                for j in neighborhood:
                    if i != j:
                        ml.append((i, j))

            for other_neighborhood in self.neighborhoods:
                if neighborhood != other_neighborhood:
                    for i in neighborhood:
                        for j in other_neighborhood:
                            cl.append((i, j))
        return ml, cl
