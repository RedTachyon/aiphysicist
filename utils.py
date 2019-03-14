import numpy as np


def get_n_most_frequent(a: np.ndarray, n: int) -> np.ndarray:
    """
    Finds n most frequent indices.

    Args:
        a: array
        n: how many indices to return

    Returns:
        idx: array of ints denoting indices

    """

    (vals, cts) = np.unique(a, return_counts=True)
    idx = cts.argsort()[::-1][:n]  # most frequent indices
    return idx

