import clustbench
from skquery.pairwise import RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC
from skquery.oracle import MLCLOracle
from active_semi_clustering import COPKMeans
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

    # Test all implemented algorithms
    for strat in tqdm([RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC]):
        qs = strat()
        if strat == NPU:
            qs = strat(clusterer=COPKMeans(n_clusters=dataset.n_clusters[0]))
        t1 = time()
        constraints = qs.fit(dataset.data, MLCLOracle(truth=labels),
                             n_clusters=dataset.n_clusters[0])
        print(f"\n{strat.__name__} : {time() - t1} s")
        assert len(constraints) > 0

    # Test all implemented algorithms
    for strat in tqdm([FFQS, MinMax, NPU]):
        constraints = {"ml": [], "cl": []}
        nbhds = None
        t1 = time()
        for i in range(2):
            algo = COPKMeans(n_clusters=dataset.n_clusters[0])
            algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])

            qs = strat(neighborhoods=nbhds)

            constraints = qs.fit(dataset.data, MLCLOracle(truth=labels),
                                 partition=algo.labels_)
            assert len(constraints) > 0
            nbhds = qs.neighborhoods
        print(f"\n{strat.__name__} incremental : {time() - t1} s")
