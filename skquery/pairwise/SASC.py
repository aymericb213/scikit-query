"""
Sequential approach for Active Constraint Selection from Abin & Beigy 2014
"""
# Authors : Salma Badri, Elis Ishimwe, Aymeric Beauchamp

from ..strategy import QueryStrategy
from ..utils import BaseSVDD
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean


class SASC(QueryStrategy):
    """
    Sequential approach for Active Constraint Selection algorithm [1]_.

    Selects a subset of boundary instances with Support Vector
    Data Description (SVDD), then computes individual constraint utility
    and sequential utility of constraints.

    Parameters
    ----------
    alpha : int, default=0.5
        Proportion of constraints selected based on Assumption A.

    Attributes
    ----------
    alpha : int, default=0.5
        Proportion of constraints selected based on Assumption A.
    support_vectors : array-like
        Indexes of support vectors found through SVDD.
    p_dists_sv : array-like
        Euclidean pairwise distance matrix of support vectors.

    References
    ----------
    .. [1] Abin, A.A., Beigy, H. Active selection of
           clustering constraints: a sequential approach. 2014.
           Pattern Recognition Volume 47, 3, pp.1443-1458.
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.partition = []
        self.alpha = alpha
        self.svdd_clusters = []
        self.support_vectors = []
        self.label_svdds = []
        self.p_dists_sv = []
        self.max_sv_pw_dist = 0  # max pairwise distance between support vectors

    def fit(self, X, oracle, partition=None, n_clusters=None):
        """
        Select pairwise constraints with SASC.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        oracle : callable
            Source of background knowledge able to answer the queries.
        partition : Ignored
            Not used, present for API consistency.
        n_clusters : Ignored
            Not used, present for API consistency.

        Returns
        -------
        constraints : dict of lists
            ML and CL sequentially selected constraints.
        """
        #K = self._get_number_of_clusters(partition, n_clusters)

        #SVDD parameters : soft margin constant and kernel parameters
        C = 0.95
        rbf_sigma = np.std(pdist(X))/2

        #self._svdd_clusters(X)
        self._svdd(X, C, rbf_sigma)
        X = self._check_dataset_type(X)
        self._distance_frontiere(X)

        c_t = []
        ml = []
        cl = []
        constraints = {"ml": ml, "cl": cl}

        u_t_ijA, u_t_ijB = self._u_ind(X)
        # Assumption A
        budget_ha = int(self.alpha * oracle.budget)
        t = 0
        while t < budget_ha:
            self.u_t(c_t, u_t_ijA)
            t += 1

        # Assumption B
        t = 0
        while t < oracle.budget - budget_ha:
            self.u_t(c_t, u_t_ijB)
            t += 1

        return constraints

    def _svdd(self, X, C, sigma):
        """
        Perform Support Vector Data Description.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        C : int
            Soft margin constant.
        sigma : int
            RBF kernel scale parameter.
        """
        self.svdd = BaseSVDD(C=C, gamma=sigma, kernel='rbf', display='off')
        self.svdd.fit(X)
        #self.boundary = svdd.plot_boundary(dataset)
        self.support_vectors = self.svdd.boundary_indices
        self.center = self.svdd.center

    def _svdd_clusters(self, dataset):
        """
          Version optimisée avec numpy
          Déterminer les données à la frontière pour chaque cluster
        """
        list_clusters = []
        list_svdd = []
        self.label_svdds = []

        """
        Regroupe chaque données d'un même cluster ensemble
        """
        indice_depart = 0
        for cluster in set(self.partition):
            cluster_temporaire = dataset[np.where(self.partition == cluster)]
            list_clusters.append(cluster_temporaire)

        """
          Calculer le svdd pour chaque cluster:
            pour chaque cluster on applique le svdd ( svdd.fit(X)):
                On récuprére les frontières via des indices (svd.boundary_indices)       
        """
        for indice in set(self.partition):
            X = np.array(list_clusters[indice])
            svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')
            # fit the SVDD model
            svdd.fit(X)
            # predict the label
            # y_predict = svdd.predict(X)

            if indice > 0:
                indice_depart += len(list_clusters[indice - 1])
                list_svdd.append(list(np.array(svdd.boundary_indices) + indice_depart))
            else:
                list_svdd.append(svdd.boundary_indices)

            nombre_frontiere_cluster = len(list_svdd[indice])
            self.label_svdds.extend([indice] * nombre_frontiere_cluster)
            self.svdd_clusters = list_svdd
            self.support_vectors = np.concatenate(list_svdd).flatten().tolist()

        print(f"\nfrontière :\n{list_svdd}\n\nLabel:\n{self.label_svdds}")

    def _distance_frontiere(self, X):
        """
        Calcul de la distance entre chaque point déterminé par le svdd du dataset avec pdist et squareform
        """
        # on récupère uniquement les données correspondant aux indices obtenus via la fonction _svdd_clusters()
        self.p_dists_sv = squareform(pdist(X[self.support_vectors, :]))
        self.max_sv_pw_dist = np.max(self.p_dists_sv)

    def _d_out_ij(self, X, i, j, samples=10):
        """
        Distance of the part of a constraint that is outside the description sphere, computed by sampling points
        along the constraint and counting how many are inside the sphere.

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        i : int
            First index of candidate constraint.
        j : int
            Second index of candidate constraint.

        Returns
        -------

        """
        sampled_points = (np.array([(1 - t) * np.asarray(X.row(i)) + t * np.asarray(X.row(j))for t in np.linspace(0, 1, samples + 2)])
                          .reshape((samples + 2, 2)))
        predictions = self.svdd.predict(sampled_points[1:-1])
        return np.sum(predictions == -1) / samples

    def _vij_out(self, X):
        N_sv = len(self.support_vectors)

        # Indices pour parcourir uniquement les paires (i, j) avec i < j
        i_idx, j_idx = np.triu_indices(N_sv, k=1)

        dij = self.p_dists_sv[i_idx, j_idx]
        dijout = dij - np.array([self._d_out_ij(X, i, j) for i, j in zip(i_idx, j_idx)])

        # Éviter la division par zéro (contraintes entre deux points identiques)
        return np.divide(dijout, dij, out=np.zeros_like(dijout), where=dij != 0), dij

    def _d_ij_ct(self, c_t, i, j):
        """
        Minimal distance between a candidate constraint and the set of previously selected constraints.

        d_{ij,C_t} = \\min_{(k,l) \\in C_t}(\\min (d(i,k)+d(j,l), d(i,l)+d(j,k)))

        Parameters
        ----------
        c_t : set of tuple
            Set of constraints already selected.
        i : int
            First index of candidate constraint.
        j : int
            Second index of candidate constraint.

        Returns
        -------
        d_ij_ct : minimal distance between (i,j) and C_t.
        """
        return min(
            min(self.p_dists_sv[i, k] + self.p_dists_sv[j, l],
                self.p_dists_sv[i, l] + self.p_dists_sv[j, k])
            for k, l in c_t
            ) if c_t else 0

    import numpy as np

    def _u_ind(self, X):
        """
        Compute individual utility of all pairwise constraints from the set of support vectors,
        based on both Assumption A and Assumption B.

        Parameters
        ----------
        X : array-like
            Instances to use for query.

        Returns
        -------
        mat_u_A, mat_u_B : (N_sv, N_sv) matrices of individual utilities using Assumption A or B.
        """
        N_sv = len(self.support_vectors)

        # Matrices d'utilité initialisée à zéro
        mat_u_A, mat_u_B = np.zeros((N_sv, N_sv)), np.zeros((N_sv, N_sv))

        # Indices pour parcourir uniquement les paires (i, j) avec i < j
        i_idx, j_idx = np.triu_indices(N_sv, k=1)

        # Distances entre toutes les paires (i, j)
        dij = self.p_dists_sv[i_idx, j_idx]

        # Distance hors du cluster
        dijout = dij - np.array([self._d_out_ij(X, i, j) for i, j in zip(i_idx, j_idx)])

        # Éviter la division par zéro
        vijout = np.divide(dijout, dij, out=np.zeros_like(dijout), where=dij != 0)

        # -------- Hypothèse A : Contraintes entre régions distinctes --------
        u_A = vijout * (1 - (dij / self.max_sv_pw_dist))

        # -------- Hypothèse B : Contraintes longues à l'intérieur d'une région --------
        u_B = (1 - vijout) * (dij / self.max_sv_pw_dist)

        # Remplissage de la matrice
        mat_u_A[i_idx, j_idx] = u_A
        mat_u_B[i_idx, j_idx] = u_B

        return mat_u_A, mat_u_B

    def u_t(self, c_t, u_ind):
        """
        La fonction u_t se charge de la sélection des contraintes en se basant sur les mesures d'utilité calculées par les fonctions u_indA et u_indB.

        """
        # Indices triés par utilité décroissante
        sorted_indices = np.dstack(np.unravel_index(np.argsort(u_ind, axis=None)[::-1], u_ind.shape))[0]
        print("Sorted", sorted_indices, u_ind)

        # Générateur qui teste chaque contrainte en commençant par la plus utile
        def constraint_generator(csts):
            for i, j in csts:
                if (i, j) not in c_t and (j, i) not in c_t:
                    yield i, j

        #Compute max distance from set of selected constraints
        max_dkl_ct = max(self._d_ij_ct(c_t, k, l) for k, l in sorted_indices)

        # Sélection des contr
        best_constraint = (None, 0)

        for i, j in constraint_generator(sorted_indices):
            print(i, j)
            if not c_t:
                c_t.append((i, j))
                return

            u_seq = u_ind[i, j]
            d_ij_ct = self._d_ij_ct(c_t, i, j)
            u_seq *= d_ij_ct / max_dkl_ct

            if u_seq >= best_constraint[1]:
                best_constraint = ((i, j), u_seq)

        c_t.append(best_constraint[0])

