from ..exceptions import MaximumQueriesExceeded


class MLCLOracle:
    def __init__(self, budget=20, truth=None):
        self.queries = 0
        self.budget = budget
        self.truth = truth

    def query(self, i, j):
        """
        Query the oracle to find out whether i and j should be must-linked
        """
        if self.queries < self.budget:
            self.queries += 1
            if self.truth is not None:
                return self.truth[i] == self.truth[j]
            return input(f"Should instances {i} and {j} be in the same cluster ? (y/n) ").strip().lower() == "y"
        else:
            raise MaximumQueriesExceeded
