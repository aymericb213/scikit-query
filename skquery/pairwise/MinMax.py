import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ..exceptions import EmptyBudgetError, QueryNotFoundError
from .FFQS import FFQS


class MinMax(FFQS):
    def _consolidate(self, X, oracle):
        skeleton = set([x for nbhd in self.neighborhoods for x in nbhd])

        unqueried_indices = set(range(X.shape[0])) - skeleton

        # Compute width σ of Gaussian kernel (20% as per the article)
        percent = 20
        assert 0 < percent <= 100
        kernel_width = np.percentile(squareform(self.pdist), percent)

        while True:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                # Compute similarity matrix with Gaussian kernel
                sims = pd.DataFrame(squareform(pdist(X,
                       lambda x, y: np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2))))))

                q_i = sims.iloc[list(unqueried_indices), list(skeleton)].max(axis=1).idxmin()

                sorted_neighborhoods = reversed(sorted(self.neighborhoods, key=lambda nbhd: sims.iloc[q_i, nbhd].max()))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                unqueried_indices.remove(q_i)

            except (EmptyBudgetError, QueryNotFoundError):
                break
