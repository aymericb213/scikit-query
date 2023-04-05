import pandas as pd
import os
import argparse
import clustbench
from selection import *
from selection.oracle import MLCLOracle
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans, PCKMeans, MPCKMeans
from sklearn.cluster import *
from sklearn.metrics import adjusted_rand_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Clustering')
    parser.add_argument('-path', type=str, help='path to clustbench data folder')
    args = parser.parse_args()

    dataset = clustbench.load_dataset("fcps", "lsun", path=args.path)
    labels = dataset.labels[0] # - 1 if args.auto else None # correspondance between clustbench and Python indexing

    algo = COPKMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)
    init_partition = algo.labels_
    print(adjusted_rand_score(labels, algo.labels_))
    
    active_qs = NPUincr()
    active_pairwise = Pairwise(algo,len(dataset))
    matrice_probabilite = active_pairwise._generer_matrice_probabilite(dataset=dataset)
    print(f'Matrice de probabilité:\n {matrice_probabilite}')
    print(f'échantillon plus claire:\n {matrice_probabilite[0]}')
    constraints = active_qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels))
    
    algo.fit(dataset.data, ml=constraints["ML"], cl=constraints["CL"])
    print(adjusted_rand_score(labels, algo.labels_))
    print(adjusted_rand_score(init_partition, algo.labels_))

