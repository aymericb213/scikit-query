import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from skquery.oracle.MLCLOracle import MaximumQueriesExceeded
from skquery import QueryStrategy


class FFQS(QueryStrategy):
    def __init__(self, neighborhoods=None):
        super().__init__()
        self.neighborhoods = [] if not neighborhoods or type(neighborhoods) != list else neighborhoods

    def fit(self, dataset, partition, oracle):
        dataset = pd.DataFrame(dataset)
        if not self.neighborhoods:
            self._explore(dataset, len(set(partition)), oracle)
        self._consolidate(dataset, oracle)

        ml, cl = self.get_constraints_from_neighborhoods()
        constraints = {"ml": ml, "cl": cl}
        return constraints

    def _explore(self, X, k, oracle):
        traversed = []
        n = X.shape[0]

        x = np.random.choice(n)
        self.neighborhoods.append([x])
        traversed.append(x)

        try:
            while len(self.neighborhoods) < k:

                max_distance = 0
                farthest = None

                for i in range(n):
                    if i not in traversed:
                        distance = dist(i, traversed, X)
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
            pass

    def _consolidate(self, X, oracle):
        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in self.neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:
            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(self.neighborhoods, key=lambda nbhd: dist(i, nbhd, X))

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


def dist(i, S, points):
    distances = np.array([euclidean(points.iloc[i, :], points.iloc[j, :]) for j in S])
    return distances.min()



