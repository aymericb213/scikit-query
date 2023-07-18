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
        self.max = 0

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

        # Assumption A
        budget_ha = int(self.alpha * oracle.budget)
        t = 0
        u_t_ij = self._u_indA(X)
        while t < budget_ha:
            self.u_t(c_t, u_t_ij, cl, ml, False)
            t += 1

        # Assumption B
        t = 0
        u_t_ij = self._u_indB(X)
        while t < oracle.budget - budget_ha:
            self.u_t(c_t, u_t_ij, cl, ml, True)
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
        # on récuprére uniquement les données correspondants aux indices obtenue via la fonction _svdd_clusters()
        self.p_dists_sv = squareform(pdist(X.iloc[self.support_vectors, :]))
        self.max = np.max(self.p_dists_sv)

    def _d_out_ij(self, X, i, j):
        """
        Distance of the part of a constraints that is outside the description sphere.

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
        f = lambda t: X.iloc[i, :] + t * (X.iloc[j, :] - X.iloc[i, :])
        candidates = [f(t) for t in np.arange(0.1, 1, 0.1) if self.is_between(X, i, j, f(t))]
        return euclidean(X.iloc[i, :], candidates[0])

    def _d_ij_ct(self, c_t, i, j):
        """
        Minimal distance between a candidate constraint and the set of previously selected constraints.

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
        distances = np.zeros(len(c_t))
        for c in range(len(c_t)):
            k, l = c_t[c]
            dik = self.p_dists_sv[i, k]
            djl = self.p_dists_sv[j, l]
            dil = self.p_dists_sv[i, l]
            djk = self.p_dists_sv[j, k]
            distances[c] = np.min([dik + djl, dil + djk])
        return np.min(distances)

    def _u_indA(self, X):
        """
        Compute individual utility of all pairwise constraints from the set of support vectors,
        based on Assumption A : constraints will be more informative if the involved points are
        close together while they are from disjointed regions.

        Parameters
        ----------
        X : array-like
            Instances to use for query.

        Returns
        -------
        u_indA : (N_sv, N_sv) matrix of individual utilities from Assumption A.

        Notes
        -----
        Cette hypothèse suggère que les contraintes qui sont plus éloignées de leur propre cluster sont plus utiles.

        Voici un résumé du fonctionnement de cette fonction :

        1. Elle initialise une matrice d'utilité vide de la taille de l'ensemble de contraintes.

        2. Elle initialise deux compteurs pour suivre le nombre de contraintes à l'intérieur et à l'extérieur du même cluster.

        3. Pour chaque paire unique de contraintes (i, j), elle effectue les étapes suivantes :

            a. Elle récupère les clusters ci et cj auxquels appartiennent respectivement i et j.

            b. Elle calcule la distance dij entre i et j.

            c. Si i et j ne sont pas dans le même cluster, elle incrémente le compteur externe et effectue les opérations suivantes :

                i. Si la distance dij est non nulle, elle calcule deux distances supplémentaires :
                min_ci et min_cj, qui représentent respectivement la distance minimale de i à tout point dans ci (à l'exception de j)
                et la distance minimale de j à tout point dans cj (à l'exception de i).
                Elle utilise ces distances pour calculer la distance de i et j à l'extérieur de leur cluster (dijout)
                et le rapport de cette distance à la distance dij (vijout). L'utilité de la contrainte (i, j) est alors calculée comme le produit de vijout
                et de 1 moins le rapport de dij à la distance maximale dans l'ensemble de contraintes.

                ii. Si la distance dij est nulle, l'utilité de la contrainte (i, j) est simplement calculée comme 1 moins le rapport de dij à la distance maximale
                  dans l'ensemble de contraintes.

            d. Si i et j sont dans le même cluster, elle incrémente le compteur interne et assigne une utilité de -1 à la contrainte (i, j).

        4. Elle renvoie la matrice d'utilité calculée.

        Ainsi, cette fonction calcule une mesure d'utilité pour chaque contrainte basée sur la distance de cette contrainte à
        son propre cluster et à d'autres clusters. Les contraintes qui sont plus éloignées de leur propre cluster
        et plus proches d'autres clusters sont considérées comme plus utiles.
        """
        N_sv = len(self.support_vectors)

        u_t_ij = np.zeros((N_sv, N_sv))

        for i in range(0, N_sv):
            for j in range(i + 1, N_sv):
                dij = self.p_dists_sv[i, j]

                dijout = dij - self._d_out_ij(X, i, j)

                vijout = dijout / dij

                u_t_ij[i, j] = vijout * (1 - (dij / self.max))
        return u_t_ij

    def _u_indB(self, X):
        """
        Calcule d'une matrice d'utilité pour un ensemble de contraintes en se basant sur l'hypothèse B.
        Cette hypothèse suggère que les contraintes qui sont plus proches de leur propre cluster sont plus utiles.

        Voici un résumé du fonctionnement de cette fonction :

            1. Elle initialise une matrice d'utilité vide de la taille de l'ensemble de contraintes.

            2. Pour chaque paire unique de contraintes (i, j), elle effectue les étapes suivantes :

            a. Elle récupère les clusters ci et cj auxquels appartiennent respectivement i et j.

            b. Elle calcule la distance dij entre i et j.

            c. Si i et j sont dans le même cluster, elle effectue les opérations suivantes :

                i. Si la distance dij est non nulle, elle calcule deux distances supplémentaires : min_ci et min_cj,
                qui représentent respectivement la distance minimale de i à tout point dans ci (à l'exception de j)
                et la distance minimale de j à tout point dans cj (à l'exception de i).
                Elle utilise ces distances pour calculer la distance de i et j à l'intérieur de leur cluster (dijout)
                et le rapport de cette distance à la distance dij (vijout).
                L'utilité de la contrainte (i, j) est alors calculée comme le produit de 1 moins vijout
                et du rapport de dij à la distance maximale dans l'ensemble de contraintes.

                ii. Si vijout est égal à 1, l'utilité de la contrainte (i, j) est simplement calculée
                comme le rapport de dij à la distance maximale dans l'ensemble de contraintes.

            d. Si i et j ne sont pas dans le même cluster, elle assigne une utilité de -1 à la contrainte (i, j).

            3. Elle renvoie la matrice d'utilité calculée.

        En résumé, cette fonction calcule une mesure d'utilité pour chaque contrainte basée sur la distance de cette contrainte à son propre cluster.
        Les contraintes qui sont plus proches de leur propre cluster sont considérées comme plus utiles.

        """
        taille = len(self.support_vectors)
        u_t_ij = np.zeros((taille, taille))

        for i in range(0, taille):
            for j in range(i + 1, taille):
                dij = self.p_dists_sv[i, j]
                dijout = dij - self._d_out_ij(X, i, j)
                vijout = dijout / dij

                u_t_ij[i, j] = (1 - vijout) * (dij / self.max)

        return u_t_ij

    def u_t(self, c_t, u_t_ij, cl, ml, link):
        """
        La fonction u_t se charge de la sélection des contraintes en se basant sur les mesures d'utilité calculées par les fonctions u_indA et u_indB.

        Voici un résumé du fonctionnement de cette fonction :

            1. Elle commence par identifier la contrainte avec la plus grande utilité dans la matrice d'utilité `u_t_ij`.

            2. Elle vérifie ensuite si cette paire de contraintes n'a pas déjà été sélectionnée. Si ce n'est pas le cas,
            elle l'ajoute à l'ensemble des contraintes sélectionnées `c_t`, ainsi qu'à la liste appropriée de contraintes de lien (`ml`)
            ou de contraintes de non-lien (`cl`), selon le paramètre `link`.

            3. Après avoir ajouté une nouvelle contrainte, elle met à jour la valeur d'utilité de cette contrainte dans la matrice d'utilité `u_t_ij`
            pour éviter de la sélectionner à nouveau.

            4. Enfin, elle met à jour les valeurs d'utilité de toutes les autres contraintes dans `u_t_ij` en fonction de leur distance à l'ensemble
            des contraintes déjà sélectionnées.
            Cette mise à jour est réalisée en multipliant l'utilité actuelle de chaque contrainte par la distance minimale entre cette contrainte
            et l'ensemble des contraintes déjà sélectionnées ( via la méthode djc_t() ), et en divisant le résultat par la distance maximale parmi toutes les contraintes.

        En résumé, cette fonction sélectionne les contraintes une par une en fonction de leur utilité,
        en veillant à mettre à jour l'utilité des contraintes restantes à chaque étape pour tenir compte de l'information déjà capturée
        par les contraintes sélectionnées.
        """
        max = self.max

        # indice de la valeur maximale
        max_ij = np.argmax(u_t_ij)

        # Récupérer les indices x et y correspondants à la matrice
        x, y = np.unravel_index(max_ij, u_t_ij.shape)
        np_c_t = np.array(c_t)
        if x != y:
            if len(c_t) > 0 and max_ij != -1:

                # vérifier si le couple (i,j) appartient au tableau
                indices = np.where((np_c_t[:, 0] == x) & (np_c_t[:, 1] == y))
                indices = np.asarray(indices)

                if indices.size == 0:
                    c_t.append((x, y))
                    if link:
                        ml.append((self.support_vectors[x], self.support_vectors[y]))
                    else:
                        cl.append((self.support_vectors[x], self.support_vectors[y]))
            else:
                c_t.append((x, y))
                if link:
                    ml.append((self.support_vectors[x], self.support_vectors[y]))
                else:
                    cl.append((self.support_vectors[x], self.support_vectors[y]))

            u_t_ij[x, y] = -1

        for i in range(0, len(self.support_vectors)):
            for j in range((i + 1), len(self.support_vectors)):

                if u_t_ij[i, j] != -1:
                    djct = self._d_ij_ct(c_t, i, j)
                    u_t_ij[i, j] = (u_t_ij[i, j] * djct) / max

    def is_between(self, X, a, b, c):
        for dim in range(X.shape[1]):
            if not min(X.iloc[a, dim], X.iloc[b, dim]) <= c[dim] <= max(X.iloc[a,dim], X.iloc[b,dim]):
                return False
        return True
