__all__ = ["EmptyBudgetError", "QueryNotFoundError"]


class EmptyBudgetError(Exception):
    """
    Exception to be raised when the number of queries made by an active method has reached the budget limit.
    """


class QueryNotFoundError(Exception):
    """
    Exception to be raised when an active method is unable to perform further queries, e.g. when all points in the dataset have been selected for query.
    """
