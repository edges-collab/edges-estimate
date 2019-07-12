==============
edges-estimate
==============

.. image:: https://img.shields.io/travis/steven-murray/edges_estimate.svg
        :target: https://travis-ci.org/steven-murray/edges_estimate

.. image:: https://readthedocs.org/projects/edges-estimate/badge/?version=latest
        :target: https://edges-estimate.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Constrain foreground and 21 cm feature parameters with EDGES data.


* Free software: GNU General Public License v3
* Documentation: https://edges-estimate.readthedocs.io.


Features
--------

* Uses yabf_ as its Bayesian framework
* Both `emcee`-based and `polychord`-based fits possible
* Range of foreground models available (eg. `LinLog`, `LogLog`, `PhysicalLin`)
* Supports arbitrary hierarchical models, and parameter dependencies.

Installation
------------
You should just be able to do `pip install .` in the top-level directory, with all
necessary dependencies automatically installed.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _yabf: https://github.com/steven-murray/yabf
