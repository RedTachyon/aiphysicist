import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from typing import Any, Optional, Callable, Tuple, List, Dict
import sympy

from utils import get_n_most_frequent, real_dl


class Theory:
    def __init__(self, pred_initializer: Callable[..., keras.Model],
                 domain_initializer: Callable[..., keras.Model],
                 pred_kwargs: Optional[Dict] = None,
                 domain_kwargs: Optional[Dict] = None):
        """
        Class that holds the information about a single theory - its predictor and domain functions.

        Args:
            pred_initializer: Callable, should return a keras model
            domain_initializer: Callable, should return a keras model
            pred_kwargs: Dict, arguments to be passed to the pred initializer
            domain_kwargs: Dict, arguments to be passed to the domain initializer
        """
        if pred_kwargs is None:
            pred_kwargs = {}
        if domain_kwargs is None:
            domain_kwargs = {}

        self.predictor = pred_initializer(**pred_kwargs)
        self.classifier = domain_initializer(**domain_kwargs)

    def predict(self, x_t: np.ndarray) -> tf.Tensor:
        """
        Based on the matrix containing a batch of observations, predict the next observation.

        The input shape is [batch, dim*T] where dim is the dimensionality of the world, and T is how many timesteps
        are used for prediction.

        Args:
            x_t: ndarray, previous observations

        Returns:
            Expected next observation
        """
        return self.predictor(x_t)  # [batch, dim]

    def domain(self, x_t: np.ndarray) -> tf.Tensor:
        """
        Based on the matrix containing a batch of observations, estimate whether this theory applies to them.

        Args:
            x_t: ndarray, previous observations

        Returns:
            Logit that the theory applies to the observations
        """
        # x_t: [batch, dim*T]
        return self.classifier(x_t)

    def evaluate(self, X: np.ndarray, Y: np.ndarray, eps: float = 10.) -> np.ndarray:
        """
        Computes the real description loss of the theory for each observation
        Args:
            X: ndarray,
            Y: ndarray,
            eps: float, DL precision

        Returns:
            Matrix of DL losses on each sample
        """
        pred = self.predict(X).numpy()
        error = np.abs(Y - pred)

        return real_dl(error, eps)

    def trainable_pred_variables(self) -> List[tf.Tensor]:
        """
        Returns a list of all trainable variables of the predictor network
        """
        return self.predictor.trainable_variables

    def trainable_domain_variables(self) -> List[tf.Tensor]:
        """
        Returns a list of all trainable variables of the domain network
        """
        return self.classifier.trainable_variables


class Hub:
    def __init__(self, initial_theories: int,
                 initializer: Callable[..., Theory]):

        self.num_theories = initial_theories
        self.theories: List[Theory] = [initializer() for _ in range(self.num_theories)]

    def propose_theories(self, X: np.ndarray, Y: np.ndarray, M_0: int) -> List[Theory]:
        """
        Implements the propose-theories routine from the paper (Alg. 5)

        Args:
            X: ndarray
            Y: ndarray
            M_0: int, number of theories to propose (must be less than or equal to self.num_theories)

        Returns:
            List of best theories

        """
        assert M_0 < self.num_theories, "M_0 must be lower or equal to the total number of theories in the hub"

        losses = []

        # evaluate each theory on each data point
        for theory in self.theories:

            loss = theory.evaluate(X, Y, eps=1.)  # (batch,)

            losses.append(loss)

        losses = np.stack(losses, 1)  # (batch, theories)

        best_ind = losses.argmin(1)  # (batch,)

        best_theories_idx = get_n_most_frequent(best_ind, M_0)  # (M_0,)

        return [self.theories[i] for i in best_theories_idx]

    def trainable_pred_variables(self) -> List[tf.Tensor]:
        """
        Returns a list of all trainable predictor variables of theories in the hub
        """
        theory_variables = [theory.trainable_pred_variables() for theory in self.theories]
        all_variables = [var for theory in theory_variables for var in theory]
        return all_variables

    def add_theories(self, theories: List[Theory], X: np.ndarray, Y: np.ndarray, eta: float):

        domain_logits = []
        for theory in theories:
            domain_logits.append(theory.domain(X).numpy())  # List[ (batch, ) ]

        domain_logits = np.stack(domain_logits, axis=1)  # (batch, theories)
        best_idx = domain_logits.argmax(axis=1)  # (batch, )  best index for each sample
        fitting_samples = []  # len: theories
        # TODO: Finish this, Algorithm 6, Page 11
        return
