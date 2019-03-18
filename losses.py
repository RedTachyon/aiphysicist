from typing import Callable

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

     Args:
         r: tensor of values
         eps: precision

     Returns:
         description length
     """

    dl_tensor = (0.5 / np.log(2)) * tf.math.log(1 + tf.square(tf.divide(r, eps)))

    return tf.reduce_sum(dl_tensor)


def generalized_mean_loss(hub: Hub, X: np.ndarray, Y: np.ndarray, loss: Callable[[tf.Tensor, np.ndarray], float],
                          gamma: float = -1.) -> float:
    """
    Computes the generalized mean loss introduced in the paper, Eq (2),

    Args:
        hub:
        X:
        Y:
        loss:
        gamma:

    Returns:

    """
    theory_losses = []

    for theory in hub.theories:
        pred = theory.predict(X)  # [batch, dim]
        theory_loss = loss(pred, Y) ** gamma  # [batch, ]
        theory_losses.append(theory_loss)

    full_loss = tf.add_n(theory_losses)  # [batch, ]
    full_loss = full_loss / len(theory_losses)
    full_loss = full_loss ** (1 / gamma)

    full_loss = tf.reduce_sum(full_loss, axis=0)

    return full_loss


def theory_dl_loss(theory: Theory, X: np.ndarray, Y: np.ndarray, eps: float):
    """
    Computes DL(T) + DL(errors)

    Defined page 12?, Cor 1.1
    Doesn't actually use weights directly I think

    Args:
        theory:
        X:
        Y:
        eps:

    Returns:

    """

    # absolutely has to be differentiable
    # TODO: NEXT, DL loss l_dl_eps like on page 8
    
