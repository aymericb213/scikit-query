"""
Sequential approach for Active Constraint Selection from Abin & Beigy 2014
"""
# Authors : Salma Badri, Elis Ishimwe

from skquery import QueryStrategy
from skquery.src.BaseSVDD import BaseSVDD
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class Sequential(QueryStrategy):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.labels = []
        self.svdd_clusters = []
        self.svdd_cluster = []
        self.label_svdds = []
        self.distance_svdd = []
        self.max = 0

    def _svdd_clusters(self):
        """
          Version  Optimisée avec numpy
          Déterminer les données à la frontière pour chaque cluster
        """
        dataset = self.dataset
        list_clusters = []
        list_svdd = []
        self.label_svdds = []

        """
        Regroupe chaque données d'un même cluster ensemble
        """
        indice_depart = 0
        for cluster in range(1, int(dataset.n_clusters) + 1):
            cluster_temporaire = dataset.data[np.where(self.labels == cluster)]
            list_clusters.append(cluster_temporaire)

        """
          Calculer le svdd pour chaque cluster:
            pour chaque cluster on applique le svdd ( svdd.fit(X)):
                On récuprére les frontières via des indices (svd.boundary_indices)       

        """
        for indice in range(int(dataset.n_clusters)):
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
            self.svdd_cluster = np.concatenate(list_svdd).flatten().tolist()

        print(f"\nfrontière :\n{list_svdd}\n\nLabel:\n{self.label_svdds}")

    def _distance_dataset(self):
        """
        Calcul de la distance entre chaque point du dataset avec pdist et squareform
        """
        dataset = self.dataset
        distances = pdist(dataset.data)
        dist_matrix = squareform(distances)
        print(dist_matrix)
        print(f'Min : => {np.amin(dist_matrix[dist_matrix != 0])}  \nMax : => {np.max(dist_matrix)}')

    def _distance_frontiere(self):
        """
        Calcul de la distance entre chaque point déterminé par le svdd du dataset avec pdist et squareform
        """
        dataset = self.dataset
        list_svdd = self.svdd_clusters
        # Transformer la liste multidimensionnelle en liste simple
        flat_list_svdd_indices = np.concatenate(list_svdd).flatten().tolist()
        # on récuprére uniquement les données correspondants aux indices obtenue via la fonction _svdd_clusters()
        data_svdd_boundary = dataset.data[flat_list_svdd_indices]
        distances_svdd = pdist(data_svdd_boundary)
        dist_matrix_svdd = squareform(distances_svdd)
        self.distance_svdd = dist_matrix_svdd
        self.max = np.max(dist_matrix_svdd)

    def fit(self, labels, oracle):
        """
         Prend en paramètre un label et un oracle
        """
        self.labels = np.array(labels)
        self.labels = self.labels + 1
        self._svdd_clusters()
        self._distance_frontiere()

        u_t_ij = self.u_indA()
        c_t = []
        ml = []
        cl = []

        contraintes = {"ml": ml, "cl": cl}

        # Hypothèse A
        nombre_question = oracle.budget
        t = 0

        while t < nombre_question:
            self.u_t(c_t, u_t_ij, cl, ml, False)
            t = t + 1

        # Hypothèse B
        t = 0
        u_t_ij = self.u_indB()
        while t < nombre_question:
            self.u_t(c_t, u_t_ij, cl, ml, True)
            t = t + 1

        print(f'\n\n\nImplémentation, Contrainte séquentielle (Ahmad Ali Abinn, Hamid Beigy):\n\n {contraintes}')
        return contraintes

    # ////////////////////////////////////////////////////////////////////////////////////////////////
    """
        Soit deux point jupter et saturne de région disjoint
        Problèmatique comment déterminer la distance entre ces points à l'exterieurs de
        leur région :
            a ) Déterminer le point appartenant à la même région que jupiter ayant la plus court distance par rapport à 
            saturne :
                1) Une fois ce point déterminer s'assurer que la distance entre ce point et jupiter n'est pas plus importante 
                   que celle entre  jupiter et saturne.

                   Dans le cas contraire retourner 0 


        Calcul de la distance en dehors de la région :
            Prend en paramètre le cluster de recherche du point minimal par rapport à jupiter , la position de jupiter et enfin celle de saturne 

            Renvoi la distance entre le point minimal déterminé et jupiter si cette distance est inférieur à celle de jupiter par rapport à saturne 
            Sinon 0
    """

    def d_out_ij(self, indice_cluster, indice_donnee1, indice_donnee2):

        svdds = self.svdd_clusters
        distance_svdd = self.distance_svdd
        # distance jupiter et saturne
        distance_donnee1_donnee2 = distance_svdd[indice_donnee1, indice_donnee2]
        distance = []
        distance_indice = []

        for point_cluster in range(0, len(svdds[indice_cluster])):
            # on ne considère que les point du cluster dont la distance est inférieur à celle entre jupiter et saturne
            if point_cluster != indice_donnee1 and distance_svdd[point_cluster, indice_donnee2] < distance_donnee1_donnee2:
                distance.append(distance_svdd[point_cluster, indice_donnee2])
                distance_indice.append(point_cluster)

        if len(distance) > 0:
            indice = distance_indice[np.argmin(distance)]
            distance_retirable = distance_svdd[indice_donnee1, indice]

            # On ne prend en compte qu'une distance inférieur à celle de saturne et jupiter
            if distance_retirable < distance_donnee1_donnee2:
                return distance_retirable
            else:
                return 0.0

        else:
            return 0.0

        # ////////////////////////////////////////////////////////////////////////////////////////////////

    """
    La fonction djc_t , calcule la distance minimale entre une contrainte candidate (i, j) et un ensemble de contraintes Ct déjà sélectionnées. 

    Voici un résumé de la fonction :

    1. Elle récupère une mesure de distance spécifique à partir de l'objet courant.
    2. Elle extrait la troisième valeur de chaque sous-ensemble dans Ct.
    3. Elle trouve et récupère la valeur minimale parmi ces troisièmes valeurs.
    4. Elle récupère tous les sous-ensembles de Ct qui contiennent cette valeur minimale.
    5. Elle prend les deux premiers de ces sous-ensembles pour calculer quatre distances 
    différentes en utilisant la mesure de distance récupérée à la première étape et 
    les indices i, j, k et l.
    6. Elle construit un tableau contenant deux valeurs, chacune étant la somme de deux distances 
    calculées à l'étape précédente.
    7. Elle renvoie la valeur minimale de ce tableau, ce qui représenter la distance minimale entre 
    la contrainte candidate (i, j) et l'ensemble de contraintes Ct déjà sélectionnées.

    """

    def djc_t(self, c_t, i, j):

        distance_svdd = self.distance_svdd
        # Récupération des 3ème valeurs de chaque sous-tableau
        troisiemes = [x[2] for x in c_t]

        # Récupération de l'indice du minimum des 3ème valeurs
        min_idx = np.argmin(troisiemes)

        # Récupération du minimum des 3ème valeurs
        min_val = troisiemes[min_idx]

        # Récupération des sous-tableaux contenant le minimum
        sous_tableaux = c_t[troisiemes == float(min_val)]
        k = sous_tableaux[0]
        l = sous_tableaux[1]
        dik = distance_svdd[i, k]
        djl = distance_svdd[j, l]
        dil = distance_svdd[i, l]
        djk = distance_svdd[j, k]
        tableau_min = [(dik + djl), (dil + djk)]
        return np.min(tableau_min)

    # //////////////////////////////////////////////////////////////////////////////////////
    """
    calcule d'une matrice d'utilité pour un ensemble de contraintes en se basant sur l'hypothèse A. 
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

    def u_indA(self):

        svdds = self.svdd_cluster
        label_svdd = self.label_svdds
        mat_distance = self.distance_svdd
        max = self.max
        taille = len(svdds)

        u_t_ij = np.zeros((taille, taille))

        cpt_out = 0
        cpt_in = 0
        for i in range(0, taille):
            for j in range((i + 1), taille):

                ci = label_svdd[i]
                cj = label_svdd[j]
                dij = mat_distance[i, j]
                dijout = dij
                vijout = 0.0
                # Vérifier si i et j ne sont pas dans le même cluster
                if not (label_svdd[i] == label_svdd[j]):
                    cpt_out += 1
                else:
                    cpt_in += 1

                if not (label_svdd[i] == label_svdd[j]):

                    if dij != 0:

                        min_ci = self.d_out_ij(ci, i, j)
                        min_cj = self.d_out_ij(cj, j, i)

                        dijout -= min_ci
                        dijout -= min_cj
                        vijout = dijout / dij

                        u_t_ij[i, j] = vijout * (1 - (dij / max))

                    else:
                        u_t_ij[i, j] = (1 - (dij / max))

                else:
                    u_t_ij[i, j] = -1

        return u_t_ij

    # ////////////////////////////////////////////////////////////////////////////////////

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

    def u_indB(self):

        svdds = self.svdd_cluster
        label_svdd = self.label_svdds
        distance_svdd = self.distance_svdd
        max = self.max
        taille = len(svdds)
        u_t_ij = np.zeros((taille, taille))

        for i in range(0, taille):
            for j in range((i + 1), taille):

                ci = label_svdd[i]
                cj = label_svdd[j]
                dij = distance_svdd[i, j]
                dijout = dij
                vijout = 0.0

                # Vérifier si i et j ne sont pas dans le même cluster
                if label_svdd[i] == label_svdd[j]:

                    if dij != 0:
                        min_ci = self.d_out_ij(ci, i, j)
                        min_cj = self.d_out_ij(cj, j, i)

                        dijout -= min_ci
                        dijout -= min_cj
                        vijout = dijout / dij

                    if vijout != 1:
                        u_t_ij[i, j] = (1 - vijout) * (dij / max)

                    else:
                        u_t_ij[i, j] = (dij / max)


                else:
                    u_t_ij[i, j] = -1.0

        return u_t_ij

    # //////////////////////////////////////////////////////////////////////////////

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

    def u_t(self, c_t, u_t_ij, cl, ml, link):

        svdds = self.svdd_cluster
        max = self.max
        distance_svdd = self.distance_svdd
        taille = len(svdds)

        # indice de la valeur maximale
        max_ij = np.argmax(u_t_ij)

        # Récupérer les indices x et y correspondants à la matrice
        x, y = np.unravel_index(max_ij, u_t_ij.shape)
        dij = distance_svdd[x, y]
        np_c_t = np.array(c_t)
        if x != y:
            if len(c_t) > 0 and max_ij != -1:

                # vérifier si le couple (i,j) appartient au tableau
                indices = np.where((np_c_t[:, 0] == x) & (np_c_t[:, 1] == y))
                indices = np.asarray(indices)

                if indices.size == 0:
                    c_t.append([x, y, dij])
                    if link:
                        ml.append([svdds[x], svdds[y]])
                    else:
                        cl.append([svdds[x], svdds[y]])


            else:
                c_t.append([x, y, dij])
                if link:
                    ml.append([svdds[x], svdds[y]])
                else:
                    cl.append([svdds[x], svdds[y]])

            u_t_ij[x, y] = -1

        for i in range(0, taille):
            for j in range((i + 1), taille):

                if u_t_ij[i, j] != -1:
                    djct = self.djc_t(c_t, i, j)
                    u_t_ij[i, j] = (u_t_ij[i, j] * djct) / max
