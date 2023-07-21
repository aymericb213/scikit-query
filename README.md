[![Documentation Status](https://readthedocs.org/projects/scikit-query/badge/?version=latest)](https://scikit-query.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/scikit-query)](https://pypi.org/project/scikit-query)
[![Python](https://img.shields.io/pypi/pyversions/scikit-query)]()
[![codecov](https://codecov.io/github/aymericb213/scikit-query/branch/main/graph/badge.svg?token=ZU4OEZKSP9)](https://codecov.io/github/aymericb213/scikit-query)
[![license](https://img.shields.io/pypi/l/scikit-query)](https://choosealicense.com/licenses/bsd-3-clause)
[![Downloads](https://static.pepy.tech/badge/scikit-query)](https://pepy.tech/project/scikit-query)

# scikit-query

Clustering aims to group data into clusters without the help of labels, unlike classification algorithms. 
A well-known shortcoming of clustering algorithms is that they rely on an objective function geared toward 
specific types of clusters (convex, dense, well-separated), and hyperparameters that are hard to tune.
Semi-supervised clustering mitigates these problems by injecting background knowledge in order to guide the clustering.
Active clustering algorithms analyze the data to select interesting points to ask the user about, generating constraints
that allow fast convergence towards a user-specified partition.

**scikit-query** is a library of active query strategies for constrained clustering inspired by [scikit-learn](https://scikit-learn.org)
and the now inactive [active-semi-supervised-clustering](https://github.com/datamole-ai/active-semi-supervised-clustering) library by Jakub Švehla.

It is focused on algorithm-agnostic query strategies, 
i.e. methods that do not rely on a particular clustering algorithm. 
From an input dataset, they produce a set of constraints by making insightful queries to an oracle.
A variant for incremental constrained clustering is provided for applicable algorithms,
taking a data partition into account. 

In typical *scikit* way, the library is used by instanciating a class and using its *fit* method.

``` python
from skquery.pairwise import AIPC
from skquery.oracle import MLCLOracle

qs = AIPC()
oracle = MLCLOracle(truth=labels, budget=10)
constraints = qs.fit(dataset, oracle)
```

## Constraints

**Must-link** and **cannot-link** (ML/CL) constraints, also referred to as pairwise constraints,
establish a relation between two data points : they must be in the same cluster (must-link)
or in separate clusters (cannot-link). These are most widely studied constraints for clustering.

**Triplet** constraints, sometimes called relative constraints, define the relationship between 
three data points : a reference point *a*, a positive point *p* and a negative point *n*.
The positive point *p* is assumed to be more similar to *a* than *n* is. Formally, it is expressed as follows:

```math
y_a = y_n \implies y_a = y_p
y_a \neq y_p \implies y_a \neq y_n
```

Querying a triplet constraint *(i,j,k)* amounts to asking the user : "Is *i* more similar to *j* than to *k* ?"
The answer to the query will determine the roles of *j* and *k* in the constraint. Indeed, "no"
would mean that *j* corresponds to the negative point *n*, and *k* corresponds to *p*, while "yes"
would mean the reverse.

## Algorithms

| Algorithm       | Description                            | Constraint type | Works in incremental setting ? | Source                                                                                  | Date |
|-----------------|----------------------------------------|-----------------|--------------------------------|-----------------------------------------------------------------------------------------|------|
| Random sampling |                                        | ML/CL, triplet  | :heavy_check_mark:             |                                                                                         |      |
| FFQS            | Neighborhood-based                     | ML/CL           | :heavy_check_mark:             | [Basu et al.](https://epubs.siam.org/doi/10.1137/1.9781611972740.31)                    | 2004   |
| MMFFQS (MinMax) | Neighborhood-based, similarity         | ML/CL           | :heavy_check_mark:             | [Mallapragada et al.](https://ieeexplore.ieee.org/document/4761792)                     | 2008                                                                 |
| NPU             | Neighborhood-based, information theory | ML/CL           | :heavy_check_mark:             | [Xiong et al.](https://dl.acm.org/doi/10.1109/TKDE.2013.22)                             | 2013                                                                 |
| SASC            | SVDD, greedy approach                  | ML/CL           |                                | [Abin & Beigy](https://www.sciencedirect.com/science/article/abs/pii/S0031320313004068) | 2014                                                                 |
| AIPC            | Fuzzy clustering, information theory   | ML/CL           |                                | [Zhang et al.](https://ieeexplore.ieee.org/document/8740960)                            | 2019                                                                                    |

## Dependencies

scikit-query is developed on Python >= 3.10, and requires the following libraries :
- pandas>=2.0.1
- matplotlib>=3.7.1
- numpy>=1.24.3
- scikit-learn>=1.2.2
- cvxopt>=1.3.1
- scikit-fuzzy>=0.4.2
- scipy>=1.10.1
- plotly>=5.14.1

## Contributors

FFQS, MinMax and NPU are based upon Jakub Švehla's implementation. 
Other algorithms have been implemented by Aymeric Beauchamp or his students from the University of Orléans :
- Salma Badri, Elis Ishimwe, Brice Jacquesson, Matthéo Pailler (2023)