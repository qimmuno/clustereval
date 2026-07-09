import pytest

from clustereval import (
    homogeneity_score,
    parsimony_score,
    normalized_purity_score,
    normalized_inverse_purity_score,
)


def test_homogeneity_is_one_for_single_class_reference():
    labels_true = [1, 1, 1, 1]
    labels_predicted = [0, 1, 2, 3]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(0.0)


def test_matching_partition():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 1, 2, 2, 3, 3]

    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)


def test_single_predicted_cluster():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 1, 1, 1, 1, 1]

    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)

def test_singleton_clusters():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 2, 3, 4, 5, 6]

    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(0.0)
