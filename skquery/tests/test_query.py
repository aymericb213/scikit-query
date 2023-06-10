import clustbench
from skquery.pairwise import RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC
from skquery.oracle import MLCLOracle
from time import time
from tqdm import tqdm
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ModuleNotFoundError:
    pass
from sklearn.cluster import KMeans


def test_query():
    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    algo = KMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)

    # Test all implemented algorithms
    for strat in [RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC]:
        qs = strat()
        constraints = qs.fit(dataset.data, MLCLOracle(truth=labels),
                             partition=algo.labels_, n_clusters=dataset.n_clusters[0])
        assert len(constraints) > 0


def timing():
    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    algo = KMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)

    for strat in tqdm([RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC]):
        qs = strat()
        t1 = time()
        qs.fit(dataset.data, MLCLOracle(truth=labels),
               partition=algo.labels_, n_clusters=dataset.n_clusters[0])
        print(f"{strat.__name__} : {time() - t1} seconds to fit")
