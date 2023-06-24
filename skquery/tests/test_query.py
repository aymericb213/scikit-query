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


def test_query():
    algorithms = [RandomMLCL, FFQS, MinMax, NPU, AIPC, SASC]
    incr_algos = [RandomMLCL, FFQS, MinMax, NPU]

    dataset = clustbench.load_dataset("fcps", "lsun", path="clustering-data-v1")
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    budget = 10
    # Test all implemented algorithms
    for strat in tqdm(algorithms):
        qs = strat()
        if strat == NPU:
            qs = strat(clusterer=COPKMeans(n_clusters=dataset.n_clusters[0]))
        t1 = time()
        constraints = qs.fit(dataset.data, MLCLOracle(truth=labels, budget=budget),
                             n_clusters=dataset.n_clusters[0])
        print(f"\n{strat.__name__} : {time() - t1} s")
        assert len(constraints) > 0

    # Tests in incremental setting
    n_iter = 2
    for strat in tqdm(incr_algos):
        constraints = {"ml": [], "cl": []}
        nbhds = None
        pdists = None
        sims = None
        t_qs = []
        for i in range(n_iter):
            algo = COPKMeans(n_clusters=dataset.n_clusters[0])
            algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])

            if strat == RandomMLCL:
                qs = strat()
            else:
                qs = strat(neighborhoods=nbhds)
            t1 = time()
            constraints = qs.fit(dataset.data, MLCLOracle(truth=labels, budget=budget),
                                 partition=algo.labels_, pdist=pdists)
            t_qs.append(time() - t1)

            assert len(constraints) > 0
            if strat in [FFQS, MinMax, NPU]:
                nbhds = qs.neighborhoods
            if strat in [FFQS, MinMax]:
                pdists = qs.pdist
        #TODO: too much clutter, make tests for each alg
        print(f"\n{strat.__name__} incremental : {sum(t_qs)} s (breakdown: {t_qs})")
