import pandas as pd


class QueryStrategy:

    def __init__(self):
        pass

    def _check_dataset_type(self, X):
        return pd.DataFrame(X)

