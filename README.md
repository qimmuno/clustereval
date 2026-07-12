[![License](https://img.shields.io/pypi/l/clustereval.svg)](https://github.com/andim/clustereval/blob/master/LICENSE)
[![Latest release](https://img.shields.io/pypi/v/clustereval.svg)](https://pypi.python.org/pypi/clustereval)
[![Documentation Status](https://readthedocs.org/projects/clustereval/badge/?version=latest)](https://clustereval.readthedocs.io/en/latest/?badge=latest)

# ClusterEval: External Clustering Validation by the Homogeneity-Parsimony Trade-Off

ClusterEval is a lightweight software package for external clustering validation.


It provides reference implementations for the homogeneity-parsimony scores proposed in [Tiffeau-Mayer 2026](TBD). Evaluation of clustering quality on these two objectives provides a unified framework for assessing clustering agreement with ground truth class labels.

The scores are implemented to be compatible with evaluation metrics defined in [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)'s `sklearn.metrics` to allow easy replacement within existing pipelines.

## Installation

ClusterEval can be installed via pip:

`pip install clustereval`

The package depends on `numpy` and `scikit-learn`.

## Documentation and examples

API documentation is hosted on [readthedocs](https://clustereval.readthedocs.io/en/latest/?badge=latest).
You can also create a local copy of the API documentation by running:

```bash
make html
```

in the docs folder.
