import clustbench
from skquery.pairwise import Random, FFQS, MinMax, NPUincr, AIPC, SASC
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
    for strat in [Random, FFQS, MinMax, NPUincr, AIPC, SASC]:
        qs = strat(dataset.n_clusters[0])
        qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=10))


def timing():
    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    algo = KMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)

    for strat in tqdm([Random, FFQS, MinMax, NPUincr, AIPC, SASC]):
        qs = strat(dataset.n_clusters[0])
        t1 = time()
        qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=10))
        print(f"{strat.__name__} : {time() - t1} seconds to fit")
