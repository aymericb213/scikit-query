# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting with negataive samples

"""
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
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
    for  indice in range ( int(dataset.n_clusters)):
       list_clusters.append([])
       list_svdd.append([])


    
    for  indice in range ( len(dataset.data)):
       numero_cluster = dataset.labels[0][indice]
       list_clusters[numero_cluster-1].append(dataset.data[indice]) 
 
    
    for indice in range (int(dataset.n_clusters)):
       X = np.array(list_clusters[indice])
       svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')
       # fit the SVDD model
       svdd.fit(X)

       # predict the label
       y_predict = svdd.predict(X)

       list_svdd[indice].append(svdd.boundary_indices) 
       
       
        
    print(f"\nfrontière :\n{list_svdd}")
    return list_svdd

svdd_clusters()   
    


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

