import numpy as np


def int_dl(m: int) -> float:
    """
    Computes the integer description length

    Args:
        m: integer

    Returns:
        description length

    """
    return np.log2(1 + np.abs(m))


def rational_dl(m: int, n: int) -> float:
    """
    Computes the rational description length

    Args:
        m: numerator
        n: denominator

    Returns:
        description length
    """
    return np.log2((1 + np.abs(m)) * n)


def real_dl(r: np.ndarray, eps: float) -> np.ndarray:
    """
    Computes the real (as in, real number) description length

    Args:
        r: value
        eps: precision

    Returns:
        description length
    """
    return 0.5 * np.log2(1 + (r / eps) ** 2)


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
