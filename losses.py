from typing import Callable, Optional, Union, List

import numpy as np

from theory import Theory, Hub
import tensorflow as tf

from utils import tf_real_dl


def generalized_mean_loss(hub: Union[Hub, List[Theory]], X: np.ndarray, Y: np.ndarray,
                          gamma: float = -1.,
                          eps: float = 10.,
                          loss: Optional[Callable[[tf.Tensor, np.ndarray], tf.Tensor]] = None) -> tf.Tensor:
    """
    Computes the generalized mean loss introduced in the paper, Eq (2)

    Usage: loss = generalized_mean_loss(hub, X, Y, gamma=-1, eps=eps)
    Then can use gradients of loss or whatever

    Args:
        hub: Hub or list of theories, containing the theories
        X: ndarray,
        Y: ndarray,
        gamma: float, coefficient of
        eps: float, precision for the description length
        loss: float

    Returns:

    """

    if loss is None:
        loss = lambda x, y: tf_real_dl(tf.abs(x - y), eps)

    theories: List[Theory] = []

    if isinstance(hub, list):
        theories = hub
    elif isinstance(hub, Hub):
        theories = hub.theories
    else:
        raise AttributeError("The first argument should be either a hub or a list of theories")

    theory_losses = []

    for theory in theories:
        pred = theory.predict(X)  # [batch, dim]
        theory_loss = tf.add(loss(pred, Y), 1e-8)  # [batch, ]

        theory_loss = theory_loss ** gamma
        theory_losses.append(theory_loss)

    full_loss = tf.add_n(theory_losses)  # [batch, ] x M -> [batch, ]
    assert len(theory_losses) > 0
    full_loss = full_loss / len(theory_losses)

    assert not any(full_loss.numpy() == 0)

    full_loss = full_loss ** (1 / gamma)
    full_loss = tf.reduce_sum(full_loss, axis=0)  # tf.float

    return full_loss
