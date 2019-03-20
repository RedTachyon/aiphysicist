from typing import Callable, Optional

import numpy as np

from theory import Theory, Hub
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


def tf_real_dl(r: tf.Tensor, eps: float) -> tf.float32:
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


def generalized_mean_loss(hub: Hub, X: np.ndarray, Y: np.ndarray,
                          gamma: float = -1.,
                          eps: float = 10.,
                          loss: Optional[Callable[[tf.Tensor, np.ndarray], float]] = None) -> tf.Tensor:
    """
    Computes the generalized mean loss introduced in the paper, Eq (2)

    Usage: loss = generalized_mean_loss(hub, X, Y, gamma=-1, eps=eps)
    Then can use gradients of loss or whatever

    Args:
        hub:
        X:
        Y:
        gamma:
        eps:
        loss:

    Returns:

    """

    if loss is None:
        loss = lambda x, y: tf_real_dl(tf.abs(x - y), eps)

    theory_losses = []

    for theory in hub.theories:
        pred = theory.predict(X)  # [batch, dim]
        theory_loss = loss(pred, Y) ** gamma  # [batch, ]
        theory_losses.append(theory_loss)

    full_loss = tf.add_n(theory_losses)  # [batch, ] x M -> [batch, ]
    full_loss = full_loss / len(theory_losses)
    full_loss = full_loss ** (1 / gamma)
    full_loss = tf.reduce_sum(full_loss, axis=0)

    return full_loss
