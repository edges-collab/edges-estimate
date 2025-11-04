==============
edges-estimate
==============

**Constrain foreground and 21 cm feature parameters with EDGES data.**

.. image:: https://readthedocs.org/projects/edges-estimate/badge/?version=latest
        :target: https://edges-estimate.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit


### NOTE: this repo is deprecated. Most functionality is included in `edges-analysis`


Features
--------

* Uses yabf_ as its Bayesian framework
* Both ``emcee``-based and ``polychord``-based fits possible
* Range of foreground models available (eg. ``LinLog``, ``LogLog``, ``PhysicalLin``)
* Supports arbitrary hierarchical models, and parameter dependencies.

Installation
------------
You should just be able to do ``pip install .`` in the top-level directory, with all
necessary dependencies automatically installed.


.. _yabf: https://github.com/steven-murray/yabf
