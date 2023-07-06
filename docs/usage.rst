Usage
=====

This page explains how to install the library and use it in Python.

Installation
------------

The library can be installed using ``pip``:

.. code-block:: console

    pip install scikit-query

Using the library
-----------------

The library has two main modules : one containing the algorithms, and one for the oracles.
They can be imported as below :

.. code-block:: python

    from skquery.pairwise import *
    from skquery.oracle import MLCLOracle

This will allow to use the constraint selection algorithms as well
as the ``MLCLOracle`` to answer queries about pairwise constraints.

Making queries
--------------

All algorithms have a ``fit`` method taking as arguments
a matrix of *n* points having *m* features and an oracle (typically from the ``skquery.oracle`` module).
The oracle must have a ``query`` method that returns a boolean value.

.. code-block:: python

    qs = AIPC()
    oracle = MLCLOracle()
    constraints = qs.fit(dataset.data, oracle)

The oracle's ``truth`` attribute can support a ground truth labeling of the data, in which case
it will automatically answer queries based on that ground truth.
If none is provided, it will ask queries to the user through the console.

.. code-block:: python

    oracle = MLCLOracle(truth=labels)

The constraints are returned as a dictionary whose keys are
constraint types and values are lists of constraints, stored as tuples :

.. code-block:: python

    {'ml': [(81, 99), (53, 93), (66, 80), (70, 65)], 'cl': [(19, 98), (122, 56), (15, 97), (47, 71), (19, 89), (139, 72)]}

