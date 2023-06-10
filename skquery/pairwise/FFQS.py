import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from ..exceptions import MaximumQueriesExceeded
from ..strategy import QueryStrategy


class FFQS(QueryStrategy):
    def __init__(self, neighborhoods=None):
        super().__init__()
        self.partition = []
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods

    def fit(self, X, oracle, **kwargs):
        X = self._check_dataset_type(X)
        if "partition" in kwargs:
            self.partition = kwargs["partition"]

        if len(self.partition) > 0:
            K = len(set(self.partition))
        elif "n_clusters" in kwargs:
            K = kwargs["n_clusters"]
        else:
            raise ValueError("No cluster number provided")

        if 0 <= len(self.neighborhoods) < K:
            self._explore(X, K, oracle)
        self._consolidate(X, oracle)

        ml, cl = self.get_constraints_from_neighborhoods()
        constraints = {"ml": ml, "cl": cl}
        return constraints

    def _explore(self, X, k, oracle):
        traversed = []

        if not self.neighborhoods:
            x = np.random.choice(X.shape[0])
            self.neighborhoods.append([x])
            traversed.append(x)

        while len(self.neighborhoods) < k:
            try:
                max_distance = 0
                farthest = None

                for i in range(X.shape[0]):
                    if i not in traversed:
                        distance = np.array([euclidean(X.iloc[i, :], X.iloc[j, :]) for j in traversed]).min()
                        if distance > max_distance:
                            max_distance = distance
                            farthest = i

                new_neighborhood = True
                for neighborhood in self.neighborhoods:
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        new_neighborhood = False
                        break

                if new_neighborhood:
                    self.neighborhoods.append([farthest])

                traversed.append(farthest)

            except MaximumQueriesExceeded:
                break

    def _consolidate(self, X, oracle):
        neighborhoods_union = set()
        for neighborhood in self.neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(X.shape[0]):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:
            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(self.neighborhoods, key=lambda nbhd: np.array([euclidean(X.iloc[i, :], X.iloc[j, :]) for j in nbhd]).min())

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        break

                neighborhoods_union.add(i)
                remaining.remove(i)

            except MaximumQueriesExceeded:
                break

    def get_constraints_from_neighborhoods(self):
        ml = []

        for neighborhood in self.neighborhoods:
            for i in neighborhood:
                for j in neighborhood:
                    if i != j:
                        ml.append((i, j))

        cl = []
        for neighborhood in self.neighborhoods:
            for other_neighborhood in self.neighborhoods:
                if neighborhood != other_neighborhood:
                    for i in neighborhood:
                        for j in other_neighborhood:
                            cl.append((i, j))

        return ml, cl
