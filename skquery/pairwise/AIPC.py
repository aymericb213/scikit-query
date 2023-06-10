"""
 Active Informative Pairwise Constraint Formulation algorithm from Zhong et al. 2019.
"""
# Authors : Brice Jacquesson, Matthéo Pailler, Aymeric Beauchamp

import pandas as pd
from sklearn.metrics import pairwise_distances
from ..exceptions import MaximumQueriesExceeded
from ..strategy import QueryStrategy
import numpy as np
import skfuzzy as fuzz


class AIPC(QueryStrategy):

    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.partition = []
        self.epsilon = epsilon
        self.centres = None
        self.u = None
        self.d = None

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
        self.epsilon = self.epsilon * K
        self._fuzzy_cmeans(X, K)

        ml, cl = [], []
        constraints = {"ml": ml, "cl": cl}

        entropy = self._entropy(X)  # récupérer les weak samples (voir discord)
        weak_samples = []
        for i in range(len(entropy)):  # récupération des weak samples
            if entropy[i][1] > np.log(K) - self.epsilon:
                # entropy[1] = entropie
                weak_samples.append(entropy[i])
        weak_samples.sort(key=lambda x: x[1], reverse=True)  # tri par entropy

        strong_samples = self._compute_medoids()  # récupération des strong samples.

        while True:
            try:
                xweak = weak_samples.pop()
                strong_samples.sort(key=lambda x: self._symmetric_relative_entropy(xweak, x, K))
                first_strong = strong_samples[0][1]
                second_strong = strong_samples[1][1]
                weak = xweak[0]

                if oracle.query(weak, second_strong):
                    ml.append((weak, second_strong))
                else:
                    cl.append((weak, second_strong))
                    ml.append((weak, first_strong))

            except MaximumQueriesExceeded:
                break

        return constraints

    def _entropy(self, X):
        """
        return entropy : list of tuple ([x, y], entropy) where x and y are the coordinate of the point and entropy is the entropy of the point.
        """
        entropy = []
        for j in range(X.shape[0]):
            entropyX = 0
            for i in range(len(self.u)):
                u_ij = self.u[i][j]
                entropyX += u_ij * np.log(u_ij)
            entropy.append([j, -entropyX])
        return entropy

    def _compute_medoids(self):
        medoids = []  # liste de tuple (indice du centre, indice du medoid)
        for i in range(len(self.d)):
            dist_min = self.d[i][0]
            minInd = 0
            for ind, e in enumerate(self.d[i]):
                if e < dist_min:
                    dist_min = e
                    minInd = ind
            medoids.append((i, minInd))

        return medoids

    def _symmetric_relative_entropy(self, xweak, x, k):
        # a et b correspondent respectivement à la première et seconde somme de l'équation (12) de l'article.
        a = 0
        b = 0
        for i in range(k):
            u_xj = self.u[i][x[1]]  # x[1] correspond à l'indice dans la data
            u_xweak = self.u[i][xweak[0]]  # xweak[0] correspond à l'indice dans la data
            a += u_xj * (np.log((np.divide(u_xj, u_xweak))))
            b += u_xweak * (np.log((np.divide(u_xweak, u_xj))))

        return (a + b) / 2

    def _fuzzy_cmeans(self, X, k):
        fcm = fuzz.cmeans(X.T, k, 2, 0.05, maxiter=1000)
        self.centres = fcm[0]
        self.u = fcm[1]
        self.d = fcm[3]
