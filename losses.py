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


def real_dl(r: float, eps: float) -> float:
    """
    Computes the real (as in, real number) description length

    Args:
        r: value
        eps: precision

    Returns:
        description length
    """
    return 0.5 * np.log2(1 + (r / eps) ** 2)
