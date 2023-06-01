<!--- 
.. image:: https://img.shields.io/pypi/v/scikit-mine.svg
  :target: https://pypi.python.org/pypi/scikit-mine/ 
-->
.. image:: https://codecov.io/gh/remiadon/scikit-query/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/remiadon/scikit-query
<!--- 
.. image:: https://pepy.tech/badge/scikit-mine
  :target: https://pepy.tech/project/scikit-mine

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/scikit-mine/scikit-mine/HEAD?filepath=docs%2Ftutorials%2Fperiodic%2Fperiodic_canadian_tv.ipynb
-->
# scikit-query

*scikit-query* is a library of active query strategies for constrained clustering inspired by [scikit-learn](https://scikit-learn.org)
and the now inactive [active-semi-supervised-clustering](https://github.com/datamole-ai/active-semi-supervised-clustering) library by Jakub Švehla.

It is focused on algorithm-agnostic query strategies, i.e. methods that do not rely on a particular clustering algorithm.
In typical **scikit** way,
## Dependencies

scikit-query is developed on Python >= 3.10, and requires the following libraries :

- pandas~=2.0.1
- active-semi-supervised-clustering
- matplotlib~=3.7.1
- numpy~=1.24.3
- scikit-learn~=1.2.2
- cvxopt~=1.3.1
- scikit-fuzzy~=0.4.2
- scipy~=1.10.1
- plotly~=5.14.1