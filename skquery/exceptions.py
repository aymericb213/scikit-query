__all__ = ["MaximumQueriesExceeded"]


class MaximumQueriesExceeded(Exception):
    """
    Exception raised when oracle budget is depleted.
    """
