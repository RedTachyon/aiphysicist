from typing import Callable, Optional, Union, List

import numpy as np

from theory import Theory, Hub, assign_theories
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


def dominant_loss(hub: Union[Hub, List[Theory]], X: np.ndarray, Y: np.ndarray,
                  eps: float = 10.,
                  loss: Optional[Callable[[tf.Tensor, np.ndarray], tf.Tensor]] = None) -> tf.Tensor:

    if loss is None:
        loss = lambda x, y: tf_real_dl(tf.abs(x - y), eps)

    theories: List[Theory] = []

    if isinstance(hub, list):
        theories = hub
    elif isinstance(hub, Hub):
        theories = hub.theories
    else:
        raise AttributeError("The first argument should be either a hub or a list of theories")

    best_idx = assign_theories(theories, X, Y)
    loss_parts = []
    for i, theory in enumerate(theories):
        idx = np.where(best_idx == i)
        X_i = X[idx]
        Y_i = Y[idx]

        pred = theory.predict(X_i)
        theory_loss = loss(pred, Y_i)  # (partial_batch, )
        loss_parts.append(theory_loss)

    full_loss = tf.concat(loss_parts, axis=0)  # (batch, )
    full_loss = tf.reduce_sum(full_loss)

    return full_loss
