import numpy as np
from src.data.splits import make_balanced_split


def test_split_balanced_deterministic():
    targets = []
    for c in range(10):
        targets += [c] * 5000
    s1 = make_balanced_split(targets, 250, 10, seed=42, include_labeled_in_unlabeled=True)
    s2 = make_balanced_split(targets, 250, 10, seed=42, include_labeled_in_unlabeled=True)
    assert len(s1["labeled_indices"]) == 250
    counts = [(np.asarray(targets)[s1["labeled_indices"]] == c).sum() for c in range(10)]
    assert counts == [25] * 10
    assert s1["labeled_indices"] == s2["labeled_indices"]
    assert len(s1["unlabeled_indices"]) == 50000
