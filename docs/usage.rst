The basics of scikit-query
==========================

This page explains how to install the library and use it in Python.

Installation
------------

The library can be installed using ``pip`` :

.. code-block:: console

    pip install scikit-query

Using the library
-----------------

The library has modules for each kind of constraint (pairwise or triplet),
plus another one containing the oracles.
They can be imported as below :

.. code-block:: python

    from skquery.pairwise import *
    from skquery.triplet import *
    from skquery.oracle import MLCLOracle

This will allow to use the constraint selection algorithms as well
as the ``MLCLOracle`` to answer queries about pairwise constraints.

Making queries
--------------

All algorithms have a ``fit`` method taking as arguments
a matrix of *n* points having *m* features and an oracle (typically from the ``skquery.oracle`` module).
The oracle must have a ``query`` method returning a boolean.

.. code-block:: python

    qs = AIPC()
    oracle = MLCLOracle()
    constraints = qs.fit(dataset.data, oracle)

The oracle's ``truth`` attribute can support a ground truth labeling of the data,
which will be used to automatically answer queries.
If none is provided, it will ask queries to the user through the CLI.

.. code-block:: python

    oracle = MLCLOracle(truth=labels)

The constraints are returned as a dictionary of constraint types paired
with lists of selected constraints.
The table below describes how the constraint dictionary is structured.

.. csv-table:: Conventions used for constraint storage
   :header: "Type", "Key", "Constraint format"

   "Must-link",          "ml",  "(int, int)"
   "Cannot-link",        "cl",  "(int, int)"
   "Triplet",       "triplet",  "(int, int, int)"