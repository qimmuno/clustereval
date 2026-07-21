from clustereval.core import purity_score
import pytest

from clustereval import (
    homogeneity_score,
    parsimony_score,
    purity_score,
    inverse_purity_score,
    normalized_purity_score,
    normalized_inverse_purity_score,
    pair_specificity_score,
    pair_sensitivity_score,
    q_measure_score,
    completeness_score,
    conditional_entropies,
)


def test_matching_partition():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 1, 2, 2, 3, 3]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(1.0)


def test_single_predicted_cluster():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 1, 1, 1, 1, 1]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(0.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(0.0)


def test_singleton_clusters():
    labels_true = [1, 1, 2, 2, 3, 3]
    labels_predicted = [1, 2, 3, 4, 5, 6]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(0.0)

def test_fragmented_class():
    labels_true = [1, 2, 3, 4]
    labels_predicted = [0, 1, 2, 3]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(1.0)

def test_single_class_fragmented_clusters():
    labels_true = [1, 1, 1, 1]
    labels_predicted = [0, 1, 2, 3]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(0.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(0.0)

def test_single_class_matching():
    labels_true = [1, 1, 1, 1]
    labels_predicted = [2, 2, 2, 2]

    assert homogeneity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert parsimony_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert normalized_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert normalized_inverse_purity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert pair_specificity_score(labels_true, labels_predicted) == pytest.approx(1.0)
    assert pair_sensitivity_score(labels_true, labels_predicted) == pytest.approx(1.0)

    assert q_measure_score(labels_true, labels_predicted) == pytest.approx(1.0)

def test_degenerate_input():
    # Test that degenerate input (empty lists) returns ValueError
    with pytest.raises(ValueError):
        homogeneity_score([], [])
    with pytest.raises(ValueError):
        parsimony_score([], [])

    with pytest.raises(ValueError):
        purity_score([], [])
    with pytest.raises(ValueError):
        inverse_purity_score([], [])

    with pytest.raises(ValueError):
        normalized_purity_score([], [])
    with pytest.raises(ValueError):
        normalized_inverse_purity_score([], [])

    with pytest.raises(ValueError):
        pair_specificity_score([], [])
    with pytest.raises(ValueError):
        pair_sensitivity_score([], [])

    with pytest.raises(ValueError):
        q_measure_score([], [])


@pytest.mark.parametrize(
    "metric",
    [
        conditional_entropies,
        completeness_score,
        homogeneity_score,
        inverse_purity_score,
        normalized_inverse_purity_score,
        normalized_purity_score,
        parsimony_score,
        pair_sensitivity_score,
        pair_specificity_score,
        purity_score,
        q_measure_score,
    ],
)
def test_mismatched_length_input(metric):
    with pytest.raises(ValueError):
        metric([0, 0, 1], [1, 1])


@pytest.mark.parametrize(
    "metric",
    [
        homogeneity_score,
        inverse_purity_score,
        normalized_inverse_purity_score,
        normalized_purity_score,
        parsimony_score,
        pair_sensitivity_score,
        pair_specificity_score,
        purity_score,
        q_measure_score,
        conditional_entropies,
        completeness_score,
    ],
)
def test_metrics_are_invariant_to_label_renaming(metric):
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_predicted = [10, 10, 10, 20, 20, 30]

    renamed_true = [7, 7, 3, 3, 9, 9]
    renamed_predicted = [100, 100, 100, 200, 200, 300]

    assert metric(labels_true, labels_predicted) == pytest.approx(
        metric(renamed_true, renamed_predicted)
    )


@pytest.mark.parametrize(
    "metric",
    [
        conditional_entropies,
        completeness_score,
        homogeneity_score,
        inverse_purity_score,
        normalized_inverse_purity_score,
        normalized_purity_score,
        parsimony_score,
        pair_sensitivity_score,
        pair_specificity_score,
        purity_score,
        q_measure_score,
    ],
)
def test_metrics_are_invariant_to_sample_order(metric):
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_predicted = [10, 10, 10, 20, 20, 30]
    sample_permutation = [4, 2, 0, 5, 3, 1]

    permuted_true = [labels_true[index] for index in sample_permutation]
    permuted_predicted = [labels_predicted[index] for index in sample_permutation]

    assert metric(labels_true, labels_predicted) == pytest.approx(
        metric(permuted_true, permuted_predicted)
    )
