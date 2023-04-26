# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting with negataive samples

"""
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import os
import argparse
import clustbench
from selection import *
from selection.oracle import MLCLOracle
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans, PCKMeans, MPCKMeans
from sklearn.cluster import *
from sklearn.metrics import adjusted_rand_score
from selection.src.BaseSVDD import BaseSVDD

# create 100 points with 2 dimensions
n = 300
dim = 2
#X = np.r_[np.random.randn(n, dim)]

parser = argparse.ArgumentParser(description='Active Clustering')
parser.add_argument('-path', type=str, help='path to clustbench data folder')
args = parser.parse_args()

dataset = clustbench.load_dataset("fcps", "lsun", path=args.path)
labels = dataset.labels[0] # - 1 if args.auto else None # correspondance between clustbench and Python indexing

algo = COPKMeans(n_clusters=dataset.n_clusters[0])
algo.fit(dataset.data)

X = dataset.data

def svdd_clusters ():
    list_clusters = []
    list_svdd = []
    

    ### Optimisation avec np
    """
    for  indice in range ( len(dataset.data)):
       numero_cluster = dataset.labels[0][indice]
       list_clusters[numero_cluster-1].append(dataset.data[indice]) 
    """
    for cluster in range(1,int(dataset.n_clusters)+1):
       
       cluster_temporaire = dataset.data[np.where(dataset.labels[0] == cluster)]
       print(len(cluster_temporaire))
       list_clusters.append( cluster_temporaire)
       print(len(list_clusters[cluster-1]))

    """
        Calculer le svdd pour chaque cluster
    """
    for indice in range (int(dataset.n_clusters)):
        X = np.array(list_clusters[indice])
        svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')
        # fit the SVDD model
        svdd.fit(X)

        # predict the label
        y_predict = svdd.predict(X)

        list_svdd.append(svdd.boundary_indices)               
          
    print(f"\nfrontière :\n{list_svdd}")   
    return list_svdd

list_svdd = svdd_clusters() 

def distance_dataset():
   distances = pdist(dataset.data)
   dist_matrix = squareform(distances) 
   print(dist_matrix)
   print(f'Min : => {np.amin(dist_matrix[dist_matrix !=0])}  \nMax : => {np.max(dist_matrix)}') 

def distances_svdd():
    # Transformer la liste multidimensionnelle en liste simple
    flat_list_svdd_indices =  np.concatenate(list_svdd).flatten().tolist()
    data_svdd_boundary =  dataset.data[flat_list_svdd_indices]
    distances_svdd = pdist(data_svdd_boundary)
    dist_matrix_svdd = squareform(distances_svdd)
    print(dist_matrix_svdd)
    print(dist_matrix_svdd[0][1])
    print(f'Min : => {np.amin(dist_matrix_svdd[dist_matrix_svdd !=0])}  \nMax : => {np.max(dist_matrix_svdd)}') 

    """distances_svdd = pdist(list_svdd)
    dist_matrix_svdd = squareform(distances_svdd)
    print(dist_matrix_svdd)
    """

distance_dataset()

print(f"\n\n\n\n")

distances_svdd()



"""    # svdd object using rbf kernel
    svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')
    X = dataset.data

    print(dataset.labels)
"""

""""
    cluster  = []
    cluster2 = []
    cluster3 = []
    for indice in range(0,len(X)) :
        if dataset.labels[0][indice] == 1:
            cluster.append(X[indice])  
            print(cluster[indice])
        elif dataset.labels[0][indice] == 2:   
            cluster2.append(X[indice])
        else:
            cluster3.append(X[indice])




    X1 = np.array(cluster)
    X2 = np.array(cluster2)
    X3 = np.array(cluster3)

    # fit the SVDD model
    svdd.fit(X1)

    # predict the label
    y_predict = svdd.predict(X1)


    print(f"\nfrontière :\n{svdd.boundary_indices}")
    print(f"\nRadius :\n{svdd.radius}")


    # plot the boundary
    svdd.plot_boundary(X1)
    svdd.plot_distance(svdd.radius, svdd.get_distance(X1))



    svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')

    # fit the SVDD model
    svdd.fit(X2)

    # predict the label
    y_predict = svdd.predict(X2)


    print(f"\nfrontière :\n{svdd.boundary_indices}")
    print(f"\nRadius :\n{svdd.radius}")


    # plot the boundary
    svdd.plot_boundary(X2)
    svdd.plot_distance(svdd.radius, svdd.get_distance(X2))



    svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')


    # fit the SVDD model
    svdd.fit(X3)

    # predict the label
    y_predict = svdd.predict(X3)


    print(f"\nfrontière :\n{svdd.boundary_indices}")
    print(f"\nRadius :\n{svdd.radius}")


    # plot the boundary
    svdd.plot_boundary(X3)
    svdd.plot_distance(svdd.radius, svdd.get_distance(X3))

"""

