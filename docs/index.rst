.. scikit-query documentation master file, created by
   sphinx-quickstart on Wed Jul  5 17:43:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scikit-query : active query strategies for constrained clustering
========================================

Clustering aims to group data into clusters without the help of labels, unlike classification algorithms.
A well-known shortcoming of clustering algorithms is that they rely on an objective function geared toward
specific types of clusters (convex, dense, well-separated), and hyperparameters that are hard to tune.
Semi-supervised clustering mitigates these problems by injecting background knowledge in order to guide the clustering.
Active clustering algorithms analyze the data to select interesting points to ask the user about, generating constraints
that allow fast convergence towards a user-specified partition.

**scikit-query** is a library of active query strategies for constrained clustering inspired by scikit-learn
and the now inactive active-semi-supervised-clustering library by Jakub Švehla.

It is focused on algorithm-agnostic query strategies,
i.e. methods that do not rely on a particular clustering algorithm.
From an input dataset, they produce a set of constraints by making insightful queries to an oracle.
A variant for incremental constrained clustering is provided for applicable algorithms,
taking a data partition into account.


.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 1
   :caption: General information

   usage
   constraints

.. toctree::
   :maxdepth: 1
   :caption: API documentation

   api/pairwise
   api/triplet
   api/strategy
   api/oracle
   api/select
   api/exceptions
