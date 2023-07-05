from ..exceptions import EmptyBudgetError, NoAnswerError


class MLCLOracle:
    """
    Oracle for pairwise queries.

    Parameters
    ----------
    budget : int, default=10
        Maximum number of queries the oracle will answer.
    truth : array-like, default=None
        Ground truth labeling that emulates a human oracle.

    Attributes
    ----------
    queries : int
        Number of queries answered by the oracle so far.
    budget : int
        Maximum number of queries the oracle will answer.
    truth : array-like, default=None
        Ground truth labeling that emulates a human oracle.
    """
    def __init__(self, budget=10, truth=None):
        self.queries = 0
        self.budget = budget
        self.truth = truth

    def query(self, i, j):
        """
        Query the oracle to find out whether two points should be in the same cluster.

        Parameters
        ----------
        i : int
            Index of first data point.
        j : int
            Index of second data point.

        Returns
        -------
        answer : bool
            Answer to the query.

        Raises
        ------
        NoAnswerError
            If the oracle doesn't give an answer, i.e. they don't know what to answer.
        """
        if self.queries < self.budget:
            self.queries += 1
            if self.truth is not None:
                return self.truth[i] == self.truth[j]
            answer = input(f"Should instances {i} and {j} be in the same cluster ? (yes (y)/no (n)/pass (p)) ").strip().lower()
            if answer == "p":
                # "don't know" answer
                raise NoAnswerError
            else:
                return answer == "y"
        else:
            raise EmptyBudgetError
