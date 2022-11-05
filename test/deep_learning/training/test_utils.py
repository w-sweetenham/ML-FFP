import numpy as np

from src.deep_learning.training.utils import val_score_reduction, prf


def test_val_score_reduction():
    scores = [0.8, 0.7]
    assert not val_score_reduction(None, None, None, scores, mean_range=2)

    scores = [0.8, 0.9, 0.85]
    assert not val_score_reduction(None, None, None, scores, mean_range=2)

    scores = [0.8, 0.9, 0.75]
    assert val_score_reduction(None, None, None, scores, mean_range=2)


def test_prf():
    predictions = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.85, 0.15], [0.25, 0.75]])
    labels = np.array([0, 1, 1, 0, 0])
    scores = prf(predictions, labels)
    assert set(scores.keys()) == {0, 1}

    assert set(scores[0].keys()) == {'precision', 'recall', 'f-score'}
    assert np.isclose(scores[0]['precision'], 2/3, atol=10**-4)
    assert np.isclose(scores[0]['recall'], 2/3, atol=10**-4)
    assert np.isclose(scores[0]['f-score'], 2/3, atol=10**-4)

    assert set(scores[1].keys()) == {'precision', 'recall', 'f-score'}
    assert np.isclose(scores[1]['precision'], 1/2, atol=10**-4)
    assert np.isclose(scores[1]['recall'], 1/2, atol=10**-4)
    assert np.isclose(scores[1]['f-score'], 1/2, atol=10**-4)