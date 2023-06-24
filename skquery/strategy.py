import pandas as pd


class QueryStrategy:

    def __init__(self):
        pass

    def _check_dataset_type(self, X):
        return pd.DataFrame(X)

    def _get_number_of_clusters(self, **kwargs):
        if "partition" in kwargs:
            return len(set(kwargs["partition"]))
        elif "n_clusters" in kwargs:
            return kwargs["n_clusters"]
        return 0


