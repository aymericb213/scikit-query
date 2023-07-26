Constraints in clustering
=========================

Constrained clustering aims to integrate user knowledge of the data to produce
partitions that better fit its expectations. This knowledge is expressed as relations involving
the cluster labels of particular points or subsets of the data, named **constraints**.
In the following, we note *y_x* the cluster label of a point *x* belonging to the dataset.

The constraints selected by the algorithms of this library are described below.

Pairwise constraints
--------------------

**Must-link** and **cannot-link** (ML/CL) constraints, also referred to as pairwise constraints,
establish a relation between two data points : they must be in the same cluster (must-link)
or in separate clusters (cannot-link). Thanks to their simplicity and the fact that many complex
constraints can be decomposed into a set of ML/CL constraints, they are most widely studied constraints ;
most active clustering algorithms focus on selecting them efficiently.

Formally, they are expressed as an (in)equality between cluster labels :

.. math::
    y_i = y_j \quad (ML)\\
    y_i \neq y_j \quad (CL)\\

Querying a pairwise constraint between two points *i* and *j* is simply asking :
"Should *i" and *j* be in the same cluster ?". If the user answers "yes", then *(i,j)* is a
*must-link* constraint. If they answer "no", it is a *cannot-link* constraint.

Triplet constraints
-------------------

**Triplet** constraints, sometimes called relative constraints, define the relationship between
three data points : a reference point *a*, a positive point *p* and a negative point *n*.
The positive point *p* is assumed to be more similar to *a* than *n* is. Formally, it is expressed as follows:

.. math::
    y_a = y_n \implies y_a = y_p \\
    y_a \neq y_p \implies y_a \neq y_n

Querying a triplet constraint *(i,j,k)* amounts to asking the user : "Is *i* more similar to *j* than to *k* ?"
The answer to the query will determine the roles of *j* and *k* in the constraint. Indeed, "no"
would mean that *j* corresponds to the negative point *n*, and *k* corresponds to *p*, while "yes"
would mean the reverse.
