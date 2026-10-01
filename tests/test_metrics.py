import numpy as np

from surgical_bleeding.metrics import dice_iou_binary, summarize_folds


def test_empty_masks_score_as_perfect_match():
    assert dice_iou_binary(np.zeros((4, 4)), np.zeros((4, 4))) == (1.0, 1.0)


def test_sample_standard_deviation_is_used():
    mean, std = summarize_folds(np.array([1.0, 3.0]))
    assert mean == 2.0
    assert np.isclose(std, np.sqrt(2.0))

