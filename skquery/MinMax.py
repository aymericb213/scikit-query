import numpy as np
from scipy.spatial.distance import euclidean
from skquery.oracle.MLCLOracle import MaximumQueriesExceeded
from skquery import FFQS


class MinMax(FFQS):
    def _consolidate(self, X, oracle):
        n = X.shape[0]

        skeleton = set()
        for neighborhood in self.neighborhoods:
            for i in neighborhood:
                skeleton.add(i)

        remaining = set()
        for i in range(n):
            if i not in skeleton:
                remaining.add(i)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = euclidean(X.iloc[i, :], X.iloc[j, :])

        kernel_width = np.percentile(distances, 20)

        while True:
            try:
                max_similarities = np.full(n, fill_value=float('+inf'))
                for x_i in remaining:
                    max_similarities[x_i] = np.max([similarity(X.iloc[x_i, :], X.iloc[x_j, :], kernel_width) for x_j in skeleton])

                q_i = max_similarities.argmin()

                sorted_neighborhoods = reversed(sorted(self.neighborhoods, key=lambda nbhd: np.max([similarity(X.iloc[q_i, :], X.iloc[n_i, :], kernel_width) for n_i in nbhd])))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                remaining.remove(q_i)

            except MaximumQueriesExceeded:
                break


def similarity(x, y, kernel_width):
    return np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2)))
