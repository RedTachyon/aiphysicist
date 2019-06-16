import numpy as np
import tensorflow as tf


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


def tf_real_dl(r: tf.Tensor, eps: float) -> tf.Tensor:
    """
     Computes the real (as in, real number) description length of a tensor, in a tf differentiable manner

     l_dl_eps = 1/2 log_2(1 + (u/eps)^2)

     Args:
         r: tensor of values
         eps: precision

     Returns:
         description length
     """

    dl_tensor = (0.5 / np.log(2)) * tf.math.log(1 + tf.square(tf.divide(r, eps)))

    return tf.reduce_sum(dl_tensor, axis=1)
