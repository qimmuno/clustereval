import numpy as np
import sklearn.metrics.cluster
from numbers import Real
from sklearn.utils._param_validation import (
        Interval,
        validate_params,
    )

def _validate_clusterings(labels_true, labels_pred):
    """Validate the input partitions.

    Raises ValueError if the partitions have different lengths or are empty.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("The two clusterings must have the same number of samples.")
    
    if len(labels_true) == 0:
        raise ValueError("The clusterings must not be empty.")

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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    """

    _validate_clusterings(labels_true, labels_pred)

    entropy_C = _entropy(labels_true)
    entropy_K = _entropy(labels_pred)

    contingency = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, sparse=True)
    nz_val = contingency.data
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
def homogeneity_score(labels_true, labels_pred):
    """Calculate the homogeneity score for a clustering.

    Homogeneity asks whether class members :math:`C` are assigned to the same
    cluster :math:`K`. Homogeneity is defined as

    .. math::

        h(C, K) = 1 - \\frac{H(C|K)}{H(C)}
    
    Homogeneity is maximal (:math:`h = 1`) when all members of a cluster share
    the same class label, and minimal (:math:`h = 0`) when cluster assignments
    provide no information about the classes.
        
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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    
    See Also
    --------
    parsimony_score : Parsimony of clustering relative to class partition
    """
    _validate_clusterings(labels_true, labels_pred)
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
def parsimony_score(labels_true, labels_pred):
    """Calculate the parsimony score for a clustering.
    
    Parsimony asks whether the :math:`N` objects are clustered as simply as
    possible given the class :math:`C`. Formally, the parsimony score is
    defined based on the conditional entropy of clusters :math:`K` given
    classes :math:`C`:

    .. math::

        p(C, K) = 1 - \\frac{H(K|C)}{\\log(N) - H(C)}.

    Parsimony is maximal (:math:`p = 1`) when all class members share
    an identical cluster label, and minimal (:math:`p = 0`) when the
    clustering fully fragments each class. 

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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.

    See Also
    --------
    homogeneity_score : Homogeneity of clustering relative to class partition
    completeness_score : Completeness is a closely related score,  which
        depends on the entropy of the clustering under evaluation rather
        than its maximum over all possible clusterings.
    """
    _validate_clusterings(labels_true, labels_pred)
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
def completeness_score(labels_true, labels_pred):
    """Calculate the completeness score for a clustering.
    
    Completeness like parsimony penalizes clustering that fragment classes.
    However, completeness uses a normalization that depends on the entropy 
    of the specific clustering under evaluation :math:`H(K)` rather than 
    its maximum over all possible clusterings :math:`\\log(N) - H(C)`.
    Formally, completeness is defined as

    .. math::

        c(C, K) = 1 - \\frac{H(K|C)}{H(K)}

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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    
    See Also
    --------
    parsimony_score : Related score that uses a normalization based 
        on the maximum possible entropy across all possible clusterings, 
        which behaves more intuitively under refinement.
    """
    _validate_clusterings(labels_true, labels_pred)
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
    normalized ``homogeneity`` and ``parsimony`` scores:

    .. math::
        Q_\\beta(C, K) = \\frac{(1 + \\beta) h p}{\\beta h + p}
    
    As a harmonic mean, the Q-measure is sensitive to low values in either
    ``homogeneity`` or ``parsimony``. The weight parameter ``beta`` controls
    the relative importance of ``homogeneity`` vs ``parsimony``. Larger
    ``beta`` puts greater weight on ``parsimony``. The Q-measure is maximal
    (:math:`Q_\\beta = 1`) for clusterings that perfectly recover the class
    partition, and minimal (:math:`Q_\\beta = 0`) when either the clustering
    is completely uninformative about the classes or when the clustering
    fully fragments each class.
        
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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.

    See Also
    --------
    homogeneity_score : Homogeneity of clustering relative to class partition
    parsimony_score : Parsimony of clustering relative to class partition
    """
    _validate_clusterings(labels_true, labels_pred)
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

    This score is the set-matching analogue of class-conditional clustering
        entropy.

    When :math:`C` is the class partition and :math:`K` is the clustering, the
    purity score is defined as:

    .. math::
        \\mathrm{purity}(C, K) = \\frac{1}{N} \\sum_k \\max_c \\left| C_c \\cap K_k \\right|


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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.

    See Also
    --------
    normalized_purity_score : Purity normalized to [0, 1]
    """
    _validate_clusterings(labels_true, labels_pred)
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
    
    This is a min-max normalized version of the purity score,
    which is defined as:

    .. math::
        \\text{norm. purity} = \\frac{\\mathrm{purity}(C, K) - \\max_c P(c)}{1-\\max_c P(c)}

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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    """
    _validate_clusterings(labels_true, labels_pred)
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

    When :math:`C` is the class partition and :math:`K` is the clustering, the
    inverse purity score is defined as:

    .. math::
        \\mathrm{inv\\text{-}purity}(C, K) = \\frac{1}{N} \sum_c \\max_k \\left| C_c \\cap K_k \\right|
    
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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    
    See Also
    --------
    normalized_inverse_purity_score : Inverse purity normalized to [0, 1].
    """
    _validate_clusterings(labels_true, labels_pred)
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
    

    This is a min-max normalized version of the inverse purity score. Given
    the number of unique classes, $L = \\mathrm{supp}(C)$, the normalized score
    is defined as:

    .. math::
        \\text{norm. inv. purity} = \\frac{\\mathrm{inv.\\text{-}purity}(C, K) - L/N}{1-L|/N}

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

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    """
    _validate_clusterings(labels_true, labels_pred)
    inverse_purity = inverse_purity_score(labels_true, labels_pred)
    inversed_purity_min = len(set(labels_true)) / len(labels_true)
    if inversed_purity_min == 1.0:
        return 1.0
    return (inverse_purity - inversed_purity_min)/(1.0-inversed_purity_min)

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def pair_specificity_score(labels_true, labels_pred):
    """Calculate the pair specificity score for a clustering.

    Pair-based evaluation metrics are based on a binary classification of pairs
    of samples. This score is the specificity of this binary classifier,
    defined as the true negative rate.

    This score is the pair-based analogue of the homogeneity score.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    pair_specificity : float
        The pair specificity score for the clustering.

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    """
    _validate_clusterings(labels_true, labels_pred)
    (tn, fp), (fn, tp) = sklearn.metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 1.0

@validate_params(
    {
        "labels_true": ["array-like"],
        "labels_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def pair_sensitivity_score(labels_true, labels_pred):
    """Calculate the pair sensitivity score for a clustering.

    Pair-based evaluation metrics are based on a binary classification of pairs
    of samples. This score is the sensitivity of this binary classifier,
    defined as the true positive rate.

    This score is the pair-based analogue of the parsimony score.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    pair_sensitivity : float
        The pair sensitivity score for the clustering.

    Raises
    ------
    ValueError
        If the two label arrays have different lengths or are empty.
    """
    _validate_clusterings(labels_true, labels_pred)
    (tn, fp), (fn, tp) = sklearn.metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0
