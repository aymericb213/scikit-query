import pandas as pd


class QueryStrategy:
    """
    Base object inherited by implementations of
    active query strategies in the library.
    """
    def __init__(self):
        pass

    def _check_dataset_type(self, X):
        """
        Converts the input dataset into a DataFrame.

        TODO: check dataset format

        Parameters
        ----------
        X : array-like
            Instances to use for query.
        """
        return pd.DataFrame(X)

    def _get_number_of_clusters(self, partition, k):
        """
        Returns the number of clusters to use in a strategy.

        Parameters
        ----------
        partition : array-like
            Partition of the data.
        k : int
            Number of clusters to find.

        Returns
        -------
        K : int
            Number of clusters to find,
            or 0 if the number of clusters cannot be determined.
        """
        if partition is not None:
            return len(set(partition))
        elif k is not None:
            return k
        return 0

    def csts_to_file(self, constraints, filename="constraints"):
        """
        Write the contents of a constraint dictionary into a text file.

        Parameters
        ----------
        constraints : dict of list
            Constraints to write.
        filename : string, default="constraints"
            Name of the output file.
        """
        res = ""
        for key in constraints:
            for cst in constraints[key]:
                match key:
                    case "label":
                        res += f"{cst[0]}, {cst[1]}\n"
                    case "ml":
                        res += f"{cst[0]}, {cst[1]}, 1\n"
                    case "cl":
                        res += f"{cst[0]}, {cst[1]}, -1\n"
                    case "triplet":
                        res += f"{cst[0]}, {cst[1]}, {cst[2]}, 3\n"

        with open(f"{filename}.txt", "w") as file:
            file.write(res)


