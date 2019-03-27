from typing import Tuple, List, Callable

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.python.keras.activations import softmax
import numpy as np

from theory import Hub, Theory
from losses import generalized_mean_loss
from utils import real_dl


def read_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv'):
    return np.genfromtxt(path, delimiter=',')


def extract_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv', T: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    assert T in (1, 2, 3), "T has to be an integer between 1 and 3"
    data = read_data(path)
    X = data[:, :2*T]
    Y = data[:, 2*T:2*T+2]

    return X, Y


def iterative_train(theories: List[Theory],
                    X: np.ndarray, Y: np.ndarray,
                    optimizer_pred: OptimizerV2 = Adam(),
                    optimizer_domain: OptimizerV2 = Adam(),
                    K: int = 10000,
                    eps: float = 10):

    trainable_pred_variables = sum(map(lambda x: x.trainable_pred_variables(), theories), [])

    trainable_domain_variables = sum(map(lambda x: x.trainable_domain_variables(), theories), [])

    for k in range(K):  # Main training loop
        """ Can be optimized by removing the double evaluation """
        with tf.GradientTape() as tape:

            loss = generalized_mean_loss(theories, X, Y, gamma=-1, eps=eps)

        gradients = tape.gradient(loss, trainable_pred_variables)

        optimizer_pred.apply_gradients(zip(gradients, trainable_pred_variables))

        losses = []

        for theory in theories:
            theory_preds = theory.predict(X).numpy()  # (batch, dim)
            loss = real_dl(np.abs(theory_preds - Y), eps).sum(axis=1)  # (batch,)
            losses.append(loss)

        losses = np.stack(losses, axis=1)  # (batch, theories)
        best_idx = np.argmin(losses, axis=1)  # (batch, ) labels for the domain classification

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


def ddac(M: int, hub_theories: List[Theory], K: int, beta_f: float, beta_c: float, eps_0: float):
    pass
