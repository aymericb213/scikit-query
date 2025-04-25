import clustbench
from skquery.triplet import RandomTriplet
from skquery.oracle import TripletOracle
from active_semi_clustering import COPKMeans
from time import time
from tqdm import tqdm


def test_query_triplet(clustbench_path):
    algorithms = [RandomTriplet]
    incr_algos = [RandomTriplet]

    dataset = clustbench.load_dataset("other", "iris", path=clustbench_path)
    labels = dataset.labels[0] - 1  # correspondance between clustbench and Python indexing

    budget = 10
    # Test all implemented algorithms
    for strat in tqdm(algorithms):
        qs = strat()
        t1 = time()
        constraints = qs.fit(dataset.data, TripletOracle(truth=labels, budget=budget),
                             n_clusters=dataset.n_clusters[0])
        #qs.csts_to_file(constraints)
        print(f"\n{strat.__name__} : {time() - t1} s")
        assert len(constraints) > 0

    """
    # Tests in incremental setting
    n_iter = 2
    for strat in tqdm(incr_algos):
        constraints = {"triplet": []}
        t_qs = []
        for i in range(n_iter):
            algo = COPKMeans(n_clusters=dataset.n_clusters[0])
            algo.fit(dataset.data, ml=constraints["ml"], cl=constraints["cl"])

            qs = strat()
            t1 = time()
            constraints = qs.fit(dataset.data, TripletOracle(truth=labels, budget=budget),
                                 partition=algo.labels_)
            t_qs.append(time() - t1)

            assert len(constraints) > 0
        print(f"\n{strat.__name__} incremental : {sum(t_qs)} s (breakdown: {t_qs})")
    """
