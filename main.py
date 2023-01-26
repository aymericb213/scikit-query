import pandas as pd
import os
import argparse
import clustbench
from selection import *
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans, PCKMeans, MPCKMeans
from sklearn.cluster import *
from sklearn.metrics import adjusted_rand_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Clustering')
    parser.add_argument('-path', type=str, help='path to clustbench data folder')
    args = parser.parse_args()

    dataset = clustbench.load_dataset("fcps", "lsun", path=args.path)
    labels = dataset.labels[0] - 1 if args.auto else None # correspondance between clustbench and Python indexing

    algo = COPKMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)
    init_partition = algo.labels_
    print(adjusted_rand_score(labels, algo.labels_))
    
    active_qs = NPUincr()
    constraints = active_qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels))
    
    algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])
    print(adjusted_rand_score(labels, algo.labels_))
    print(adjusted_rand_score(init_partition, algo.labels_))

