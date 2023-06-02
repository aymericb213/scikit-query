[![codecov](https://codecov.io/github/aymericb213/scikit-query/branch/main/graph/badge.svg?token=ZU4OEZKSP9)](https://codecov.io/github/aymericb213/scikit-query)

# scikit-query

Clustering aims to group data into clusters without the help of labels, unlike classification algorithms. 
A well-known shortcoming of clustering algorithms is that they rely on an objective function geared toward 
specific types of clusters (convex, dense, well-separated), and hyperparameters that are hard to tune.
Semi-supervised clustering mitigates these problems by injecting background knowledge in order to guide the clustering.
Active clustering algorithms analyze the data to select interesting points to ask the user about, generating constraints
that allow fast convergence towards a user-specified partition.

**scikit-query** is a library of active query strategies for constrained clustering inspired by [scikit-learn](https://scikit-learn.org)
and the now inactive [active-semi-supervised-clustering](https://github.com/datamole-ai/active-semi-supervised-clustering) library by Jakub Švehla.

It is focused on algorithm-agnostic query strategies, i.e. methods that do not rely on a particular clustering algorithm. 
From an input dataset, they produce a set of constraints by making insightful queries to an oracle.

In typical *scikit* way, the library is used by instanciating a class and using its *fit* method.

``` python
qs = QueryStrategy()
oracle = MLCLOracle(truth=labels, budget=10)
constraints = qs.fit(dataset.data, oracle)
```

## Algorithms

- random sampling
- [FFQS](https://epubs.siam.org/doi/10.1137/1.9781611972740.31) from Basu et al. 2004
- [MinMax](https://ieeexplore.ieee.org/document/4761792) from Mallapragada et al. 2008
- [NPU](https://dl.acm.org/doi/10.1109/TKDE.2013.22) from Xiong et al. 2013. 
This is an incremental variant that doesn't rely on a constrained clustering algorithm but rather takes a partition as input and outputs a constraint set.
- [AIPC](https://ieeexplore.ieee.org/document/8740960) from Zhang et al. 2019
- [SASC](https://www.sciencedirect.com/science/article/abs/pii/S0031320313004068) from Abin & Beigy 2014

## Dependencies

scikit-query is developed on Python >= 3.10, and requires the following libraries :

- numpy~=1.24.3
- scipy~=1.10.1
- pandas~=2.0.1
- scikit-learn~=1.2.2
- scikit-fuzzy~=0.4.2
- cvxopt~=1.3.1
- matplotlib~=3.7.1
- plotly~=5.14.1

## Contributors

FFQS, MinMax and NPU are based off the original implementation of Jakub Švehla and changed for library consistency. 
Other algorithms have been implemented by Aymeric Beauchamp or his students from the University of Orléans :
- Salma Badri, Elis Ishimwe, Brice Jacquesson, Matthéo Pailler (2023)