from ..exceptions import EmptyBudgetError, NoAnswerError


class TripletOracle:
    """
    Oracle for triplet queries. Inspired from the formalization in [1].

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

    References
    ----------
    [1] Xiong, S., Pei, Y., Rosales, R., Fern, X. Z. Active Learning from Relative Comparisons,
    IEEE Transactions on Knowledge and Data Engineering, vol. 27, nᵒ 12, p. 3166‑3175, dec. 2015,
    doi: 10.1109/TKDE.2015.2462365.
    """
    def __init__(self, budget=10, truth=None):
        self.queries = 0
        self.budget = budget
        self.truth = truth

    def query(self, i, j, k):
        """
        Query the oracle on the relation between a triplet of points.

        Parameters
        ----------
        i : int
            Index of first data point, the reference.
        j : int
            Index of second data point, the assumed positive example.
        k : int
            Index of third data point, the assumed negative example.

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
                if self.truth[i] == self.truth[j] and self.truth[i] != self.truth[k]:
                    return True
                elif self.truth[i] != self.truth[j] and self.truth[i] == self.truth[k]:
                    return False
                else:
                    raise NoAnswerError
            answer = input(f"Is {i} more similar to {j} than to {k} ? (yes (y)/no (n)/pass (p)) ").strip().lower()
            if answer == "p":
                # "don't know" answer
                raise NoAnswerError
            else:
                return answer == "y"
        else:
            raise EmptyBudgetError
