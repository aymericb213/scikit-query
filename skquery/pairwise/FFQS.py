import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ..exceptions import EmptyBudgetError, QueryNotFoundError
from ..strategy import QueryStrategy


class FFQS(QueryStrategy):
    def __init__(self, neighborhoods=None):
        super().__init__()
        self.partition = []
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods
        self.pdist = pd.DataFrame()

    def fit(self, X, oracle, **kwargs):
        X = self._check_dataset_type(X)
        K = self._get_number_of_clusters(**kwargs)

        if "pdist" in kwargs and kwargs["pdist"] is not None:
            self.pdist = pd.DataFrame(kwargs["pdist"])
        else:
            # Precompute pairwise distances for farthest-first traversal
            self.pdist = pd.DataFrame(squareform(pdist(X)))

        # If K is not known, only Explore is run
        if K == 0 or 0 <= len(self.neighborhoods) < K:
            self._explore(X, K, oracle)
        # Consolidate if Explore is done and budget is not depleted
        if oracle.queries < oracle.budget:
            self._consolidate(X, oracle)

        ml, cl = self.get_constraints_from_neighborhoods()
        constraints = {"ml": ml, "cl": cl}
        return constraints

    def _explore(self, X, k, oracle):
        traversed = []
        if not self.neighborhoods:
            # Initialization
            x = np.random.choice(X.shape[0])
            self.neighborhoods.append([x])
            traversed.append(x)
        else:
            # Retrieving past traversed points from neighborhoods
            traversed = [x for nbhd in self.neighborhoods for x in nbhd]

        unqueried_indices = list(set(range(X.shape[0])) - set(traversed))

        # Run until budget runout if k is not known
        while k == 0 or len(self.neighborhoods) < k:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                # Farthest point : max distance to the set of traversed points
                # d(point, set) = min([d(point, x) for x in set])
                farthest = self.pdist.iloc[unqueried_indices, traversed].min(axis=1).idxmax()

                # Querying neighborhood of farthest point
                new_neighborhood = True
                for neighborhood in self.neighborhoods:
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        new_neighborhood = False
                        break
                if new_neighborhood:
                    self.neighborhoods.append([farthest])

                traversed.append(farthest)
                unqueried_indices.remove(farthest)

            except (EmptyBudgetError, QueryNotFoundError):
                break

    def _consolidate(self, X, oracle):
        neighborhoods_union = set([x for nbhd in self.neighborhoods for x in nbhd])

        unqueried_indices = set(range(X.shape[0])) - neighborhoods_union

        while True:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                i = np.random.choice(list(unqueried_indices))

                sorted_neighborhoods = sorted(self.neighborhoods, key=lambda nbhd: self.pdist.iloc[i, nbhd].min())

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        break

                neighborhoods_union.add(i)
                unqueried_indices.remove(i)

            except (EmptyBudgetError, QueryNotFoundError):
                break

    def get_constraints_from_neighborhoods(self):
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
