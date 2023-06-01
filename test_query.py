himport clustbench
from skquery import *
from skquery.oracle import MLCLOracle
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans


def test_query():
    # Choose a dataset from https://clustering-benchmarks.gagolewski.com/weave/data-v1.html
    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    algo = COPKMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)

    for strat in [Random, NPUincr, AIPC]:
        qs = strat(dataset.n_clusters[0])

        constraints = qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=100))
        algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])
