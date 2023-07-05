import pandas as pd


class QueryStrategy:
    """
    Base object inherited by implementations of
    active query strategies in the library.
    """

    def __init__(self):
        pass

    def _check_dataset_type(self, X):
        return pd.DataFrame(X)

    def _get_number_of_clusters(self, partition, k):
        if partition is not None:
            return len(set(partition))
        elif k is not None:
            return k
        return 0


