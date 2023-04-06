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

# svdd object using rbf kernel
svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='on')
X = dataset.dataset

# fit the SVDD model
svdd.fit(X)

# predict the label
y_predict = svdd.predict(X)

print(f"données : {X} \n len(X) = {len(X)}")
print(f"\nfrontière :\n{svdd.boundary_indices}")
print(f"\nRadius :\n{svdd.radius}")


# plot the boundary
svdd.plot_boundary(X)

# plot the distance
radius = svdd.radius
distance = svdd.get_distance(X)
print(f"\nDistance :\n{distance}")
svdd.plot_distance(radius, distance)