ClusterEval
===========

ClusterEval is a lightweight package for external clustering validation.

ClusterEval provides reference implementations for the homogeneity-parsimony
scores proposed in [Tiffeau-Mayer 2026](TBD). Evaluation of clustering quality
on these two objectives provides a unified framework for assessing clustering
agreement with ground truth class labels.

The package also provides set-matching variants of the homogeneity and
parsimony scores, called normalized purity, and normalized inverse purity.
These scores fix the definition of purity and inverse purity so that the full
range of [0, 1] is attainable. The pair-based equivalents of these scores are
pair specificity and pair sensitivity, exactly the familiar binary classifier
metrics used by the receiver operating characteristic (ROC) curve.

All scores are implemented in a way that is compatible with evaluation metrics
defined in [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)'s
`sklearn.metrics` to allow easy replacement within existing pipelines.

All metrics require non-empty label arrays of equal length and raise
``ValueError`` otherwise.

API Reference
-------------

.. automodule:: clustereval
   :members: