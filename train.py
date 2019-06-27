from typing import Tuple, List, Callable, Optional, Dict

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.activations import softmax
import numpy as np

from theory import Theory, assign_theories
from losses import generalized_mean_loss, dominant_loss


def read_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv'):
    return np.genfromtxt(path, delimiter=',')


def extract_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv', T: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    assert T in (1, 2, 3), "T has to be an integer between 1 and 3"
    data = read_data(path)
    X = data[:, :2*T]
    Y = data[:, 2*T:2*T+2]

    return X, Y


def set_precision(theories: List[Theory], X: np.ndarray, Y: np.ndarray) -> float:
    best_idx = assign_theories(theories, X, Y)
    loss_parts = []
    for i, theory in enumerate(theories):
        idx = np.where(best_idx == i)
        X_i = X[idx]
        Y_i = Y[idx]

        pred = theory.predict(X_i)
        theory_loss = tf.reduce_mean(pred - Y_i)  # (partial_batch, )
        loss_parts.append(theory_loss)

    full_loss = tf.concat(loss_parts, axis=0)  # (batch, )
    return full_loss


# First arg could be Union[List[Theory], Hub]
def iterative_train(theories: List[Theory],
                    X: np.ndarray, Y: np.ndarray,
                    loss_func: Callable[..., tf.Tensor] = generalized_mean_loss,
                    optimizer_pred: OptimizerV2 = Adam(),
                    optimizer_domain: OptimizerV2 = Adam(),
                    K: int = 10000,
                    eps: float = 10.,
                    loss_kwargs: Optional[Dict] = None):

    if loss_kwargs is None:
        loss_kwargs = {"gamma": -1, "eps": eps}

    trainable_pred_variables = sum(map(lambda x: x.trainable_pred_variables(), theories), [])

    trainable_domain_variables = sum(map(lambda x: x.trainable_domain_variables(), theories), [])
    # flag = False

    for k in range(K):  # Main training loop
        """ Can be optimized by removing the double evaluation """

        # Predictor optimization
        with tf.GradientTape() as tape:

            loss = loss_func(theories, X, Y, **loss_kwargs)
            if not k % 100:
                print("Step %d loss %.5f" % (k, loss.numpy()))

        gradients = tape.gradient(loss, trainable_pred_variables)
        optimizer_pred.apply_gradients(zip(gradients, trainable_pred_variables))

        best_idx = assign_theories(theories, X, Y, ) # (batch, ) labels for the domain classification

        with tf.GradientTape() as tape:

            domain_probs = []
            for theory in theories:
                preds = theory.domain(X)  # (batch, 1)
                domain_probs.append(preds)

            domain_probs = tf.concat(domain_probs, axis=1)  # (batch, theories)
            domain_probs = softmax(domain_probs, axis=1)  # (batch, theories)

            cce = SparseCategoricalCrossentropy()
            loss = cce(y_true=best_idx, y_pred=domain_probs)

        gradients = tape.gradient(loss, trainable_domain_variables)
        optimizer_domain.apply_gradients(zip(gradients, trainable_domain_variables))


def ddac(M: int,
         hub_theories: List[Theory],
         theory_initializer: Callable[..., Theory],
         X: np.ndarray, Y: np.ndarray,
         it_kwargs: Dict,
         K: int, beta_f: float, beta_c: float, eps_0: float):

    # it_kwargs should have optimizer_pred, optimizer_domain and K
    M_0 = len(hub_theories)  # Number of theories proposed from the hub
    num_new_theories = M - M_0

    new_theories: List[Theory] = []
    for _ in range(num_new_theories):
        new_theories.append(theory_initializer())

    theories = hub_theories + new_theories  # len = M

    eps = eps_0

    # TODO: complete DDAC

    # Harmonic training with DL loss
    for k in range(5):
        iterative_train(theories, X, Y, eps=eps, **it_kwargs)
        eps = set_precision(theories, X, Y)

    # Fine tune each theory and its domain
    for k in range(2):
        iterative_train(theories, X, Y, loss_func=dominant_loss, eps=eps, loss_kwargs={"eps": eps}, **it_kwargs)
        eps = set_precision(theories, X, Y)