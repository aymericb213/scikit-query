import clustbench
from skquery import *
from skquery.oracle import MLCLOracle
from sklearn.cluster import KMeans


def test_query():
    # Choose a dataset from https://clustering-benchmarks.gagolewski.com/weave/data-v1.html
    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    algo = KMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)

    for strat in [Random, FFQS, MinMax, NPUincr, AIPC, SASC]:
        qs = strat(dataset.n_clusters[0])
        qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=10))
