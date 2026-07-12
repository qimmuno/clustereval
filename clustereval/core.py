import numpy as np
import scipy.sparse
import sklearn.metrics.cluster
from numbers import Real
from sklearn.utils._param_validation import (
        Interval,
        validate_params,
    )


def _entropy(labels):
    """Calculate the entropy for a labeling.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        The labels.

    Returns
    -------
    entropy : float
       The entropy for a labeling.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0

    pi = np.unique(labels, return_counts=True, equal_nan=False)[1].astype(np.float64)
    pi /= np.sum(pi)

    if pi.size == 1:
        return 0.0

    return -np.sum(pi * np.log(pi))

def conditional_entropies(labels_true, labels_pred):
    """Calculate conditional entropies between two partitions.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels C to be used as a reference.

    labels_pred : array-like of shape (n_samples,)
        Cluster labels K to evaluate.

    Returns
    -------
    entropy_joint, entropy_CK, entropy_KC, entropy_C, entropy_K
    """


    entropy_C = _entropy(labels_true)
    entropy_K = _entropy(labels_pred)

    contingency = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, sparse=True)

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        # For a sparse matrix
        nzx, nzy, nz_val = scipy.sparse.find(contingency)

    nsum = np.sum(nz_val)
    entropy_joint = -np.sum(nz_val * np.log(nz_val))/nsum + np.log(nsum)

    entropy_CK = max(0, entropy_joint - entropy_K)
    entropy_KC = max(0, entropy_joint - entropy_C)
    return entropy_joint, entropy_CK, entropy_KC, entropy_C, entropy_K

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def parsimony_score(labels_true, labels_pred):
    """Calculate the parsimony score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    parsimony : float
        The parsimony score for the clustering.
    """
    entropy_joint, entropy_CK, entropy_KC, entropy_C, entropy_K = conditional_entropies(labels_true, labels_pred)
    denominator = np.log(len(labels_true))-entropy_C
    if denominator == 0:
        return 1.0
    return 1-entropy_KC/denominator

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def homogeneity_score(labels_true, labels_pred):
    """Calculate the homogeneity score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    homogeneity : float
        The homogeneity score for the clustering.
    """
    entropy_joint, entropy_CK, entropy_KC, entropy_C, entropy_K = conditional_entropies(labels_true, labels_pred)

    if entropy_C == 0:
        return 1.0
    return 1-entropy_CK/entropy_C

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def completeness_score(labels_true, labels_pred):
    """Calculate the completeness score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    completeness : float
        The completeness score for the clustering.
    """
    entropy_joint, entropy_CK, entropy_KC, entropy_C, entropy_K = conditional_entropies(labels_true, labels_pred)
    if entropy_K == 0:
        return 1.0
    return 1-entropy_KC/entropy_K

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
        "beta": [Interval(Real, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def q_measure_score(labels_true, labels_pred, *, beta=1.0):
    """Calculate the Q-measure for a clustering.

    The Q-measure is defined as the weighted harmonic mean of the 
    normalized ``homogeneity`` and ``parsimony`` scores.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.  beta : float, default=1.0
    beta : float, default=1.0
        Weight parameter controlling importance of ``homogeneity`` vs
        ``parsimony``. Larger ``beta`` puts greater weight on
        ``parsimony``. Must be >= 0.

    Returns
    -------
    q_measure : float
        The Q-measure for the clustering.

    See Also
    --------
    homogeneity_score : Homogeneity of clustering relative to class partition
    parsimony_score : Parsimony of clustering relative to class partition
    """
    h = homogeneity_score(labels_true, labels_pred)
    p = parsimony_score(labels_true, labels_pred)
    if h <= 0.0 or p <= 0.0:
        return 0.0
    return (1 + beta) * (h * p) / (beta * h + p)

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def purity_score(labels_true, labels_pred):
    """Calculate the purity score for a clustering.

    This score is the set-matching analogue of 
    class-conditional clustering entropy.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
            Cluster labels to evaluate.

    Returns
    -------
    purity : float
        The purity score for the clustering.

    See Also
    --------
    normalized_purity_score : Purity normalized to [0, 1]
    inverse_purity_score : Corresponding score equivalent to parsimony
    """
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def normalized_purity_score(labels_true, labels_pred):
    """Calculate the normalized purity score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    normalized_purity : float
        The normalized purity score for the clustering.
    """
    purity = purity_score(labels_true, labels_pred)
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred)
    purity_min = np.amax(np.sum(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    if purity_min == 1.0:
        return 1.0
    return (purity - purity_min)/(1.0-purity_min)

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def inverse_purity_score(labels_true, labels_pred):
    """Calculate the inverse purity score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    inverse_purity : float
        The inverse purity score for the clustering.
    """
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def normalized_inverse_purity_score(labels_true, labels_pred):
    """Calculate the normalized inverse purity score for a clustering.
    
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    normalized_inverse_purity : float
        The normalized inverse purity score for the clustering.
    """
    inverse_purity = inverse_purity_score(labels_true, labels_pred)
    inversed_purity_min = len(set(labels_true)) / len(labels_true)
    if inversed_purity_min == 1.0:
        return 1.0
    return (inverse_purity - inversed_purity_min)/(1.0-inversed_purity_min)
