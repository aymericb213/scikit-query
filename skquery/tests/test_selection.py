import clustbench
from time import time
from tqdm import tqdm
from active_semi_clustering import COPKMeans
from skquery.select import *


def test_selection():
    selections = [RandomSelection, NearestNeighborsSelection, EntropySelection]

    dataset = clustbench.load_dataset("other", "iris", path="clustering-data-v1")

    cc_alg = COPKMeans(n_clusters=dataset.n_clusters[0])
    cc_alg.fit(dataset.data)
    # Test all implemented algorithms
    for selection in tqdm(selections):
        select = selection(0.1)
        t1 = time()
        selected, best = select.select(dataset.data, partition=cc_alg.labels_, return_best=True)
        print(f"\n{selection.__name__} : {time() - t1} s")
        assert len(selected) > 0
