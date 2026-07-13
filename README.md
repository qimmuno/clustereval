[![License](https://img.shields.io/pypi/l/clustereval.svg)](https://github.com/andim/clustereval/blob/master/LICENSE)
[![Latest release](https://img.shields.io/pypi/v/clustereval.svg)](https://pypi.python.org/pypi/clustereval)
[![Documentation Status](https://readthedocs.org/projects/clustereval/badge/?version=latest)](https://clustereval.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/qimmuno/clustereval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/qimmuno/clustereval/actions/workflows/ci.yml)

# ClusterEval: External Clustering Validation by the Homogeneity-Parsimony Trade-Off

ClusterEval is a lightweight package for external clustering validation.

ClusterEval provides reference implementations for the homogeneity-parsimony scores proposed in [Tiffeau-Mayer 2026](TBD). Evaluation of clustering quality on these two objectives provides a unified framework for assessing clustering agreement with ground truth class labels.

The package also provides set-matching variants of the homogeneity and parsimony scores, called normalized purity, and normalized inverse purity. These scores fix the definition of purity and inverse purity so that the full range of [0, 1] is attainable. The pair-based equivalents of these scores are pair specificity and pair sensitivity, exactly the familiar binary classifier metrics used by the receiver operating characteristic (ROC) curve.

| Evaluation Approach | Criterion 1 | Criterion 2 |
| --- | --- | --- |
| Shannon Entropy | Homogeneity Score | Parsimony Score |
| Set Matching | Normalized Purity Score | Normalized Inverse Purity Score |
| Pair Counting | Pair Specificity | Pair Sensitivity |

All scores are implemented in a way that is compatible with evaluation metrics defined in [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)'s `sklearn.metrics` to allow easy replacement within existing pipelines.

## Installation

ClusterEval can be installed via pip:

`pip install clustereval`

The package depends on `numpy` and `scikit-learn`.

## Documentation and examples

API documentation is hosted on [readthedocs](https://clustereval.readthedocs.io/en/latest/?badge=latest). Jupyter example notebooks can be found in the `examples` folder.

You can create a local copy of the API documentation in the docs folder by running:

```bash
make html
```

## Support and contributing

For bug reports and enhancement requests use the [Github issue tool](http://github.com/qimmuno/clustereval/issues/new), or (even better!) open a [pull request](http://github.com/qimmuno/clustereval/pulls) with relevant changes.

When preparing a pull request, please run the testsuite using `pytest` to ensure none of the existing functionality breaks.
