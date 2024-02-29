import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ..exceptions import EmptyBudgetError, QueryNotFoundError, NoAnswerError
from .FFQS import FFQS


class MinMax(FFQS):
    """
    Min-max Farthest First Query Selection [1]_.

    FFQS with modified Consolidate step with similarity
    for constraint selection.

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
    .. [1] Mallapragada, P. K., Jin, R., Jain, A. K. Active query selection for
           semi-supervised clustering. 2008. 19th International Conference on Pattern Recognition
           (ICPR). ISBN 978-1-4244-2174-9.
    """
    def _consolidate(self, X, oracle):
        """
        Consolidate phase of MMFFQS.
        Select the point minimal similarity to its most similar point
        in the skeleton.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        """
        skeleton = set([x for nbhd in self.neighborhoods for x in nbhd])

        unqueried_indices = set(range(X.shape[0])) - skeleton - self.unknown

        # Compute width σ of Gaussian kernel (20th percentile of the pairwise distances distribution
        # as per the article)
        percent = 20
        assert 0 < percent <= 100
        # Here squareform makes square-form -> vector-form transformation
        # Gets the distribution of unique pairwise distances
        kernel_width = np.percentile(squareform(self.p_dists), percent)

        while True:
            try:
                if unqueried_indices == set():
                    raise QueryNotFoundError

                # Compute similarity matrix with Gaussian kernel
                sims = pd.DataFrame(squareform(pdist(X,
                       lambda x, y: np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2))))))

                q_i = sims.iloc[list(unqueried_indices), list(skeleton)].max(axis=1).idxmin()

                sorted_neighborhoods = reversed(sorted(self.neighborhoods, key=lambda nbhd: sims.iloc[q_i, nbhd].max()))

                try:
                    for neighborhood in sorted_neighborhoods:
                        if oracle.query(q_i, neighborhood[0]):
                            neighborhood.append(q_i)
                            break
                    skeleton.add(q_i)
                except NoAnswerError:
                    self.unknown.add(q_i)

                unqueried_indices.remove(q_i)

            except (EmptyBudgetError, QueryNotFoundError):
                break
