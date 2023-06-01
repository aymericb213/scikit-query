import pandas as pd
import argparse
import clustbench
from skquery import AIPC
from skquery.oracle import MLCLOracle
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans, PCKMeans, MPCKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import plotly.express as px
import plotly.graph_objects as go

# Choose a dataset from https://clustering-benchmarks.gagolewski.com/weave/data-v1.html
dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1-1.1.0")
labels = dataset.labels[0] - 1 # correspondance between clustbench and Python indexing

algo = COPKMeans(n_clusters=dataset.n_clusters[0])
algo.fit(dataset.data)
init_partition = algo.labels_

qs = AIPC(dataset.n_clusters[0])

constraints = qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=100))
algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])
