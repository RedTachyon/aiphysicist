import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from typing import Any, Optional, Callable, Tuple, List, Dict
import sympy

from utils import get_n_most_frequent


class Theory:
    def __init__(self, pred_initializer: Callable[..., keras.Model],
                 domain_initializer: Callable[..., keras.Model],
                 pred_kwargs: Optional[Dict] = None,
                 domain_kwargs: Optional[Dict] = None):
        """

        Args:
            pred_initializer: function that returns a model, mby neural net
            domain_initializer: as above
        """
        if pred_kwargs is None:
            pred_kwargs = {}
        if domain_kwargs is None:
            domain_kwargs = {}

        self.predictor = pred_initializer(**pred_kwargs)
        self.classifier = domain_initializer(**domain_kwargs)

    def predict(self, x_t: np.ndarray) -> tf.Tensor:
        # x_t: [batch, dim*T]
        return self.predictor(x_t)  # [batch, dim]

    def domain(self, x_t: np.ndarray) -> tf.Tensor:
        # x_t: [batch, dim*T]
        return self.classifier(x_t)

    def evaluate(self, dataset: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # TODO Apres-next
        pass

    def trainable_pred_variables(self):
        return self.predictor.trainable_variables


class Hub:
    def __init__(self, initial_theories: int,
                 initializer: Callable[..., Theory]):

        self.theories: List[Theory] = [initializer() for _ in range(initial_theories)]

    def propose_theories(self, dataset: Tuple[np.ndarray, np.ndarray], M_0: int) -> List[Theory]:
        # X, Y = dataset
        losses = []

        # evaluate each theory on each data point
        for theory in self.theories:
            # pred_Y = theory.predict(X)
            loss = theory.evaluate(dataset)  # (batch,)

            losses.append(loss)

        losses = np.stack(losses, 1)  # (batch, theories)

        best_ind = losses.argmin(1)  # (batch,)

        best_theories_idx = get_n_most_frequent(best_ind, M_0)  # (M_0,)

        return [self.theories[i] for i in best_theories_idx]

    def trainable_pred_variables(self):
        theory_variables = [theory.trainable_pred_variables() for theory in self.theories]
        all_variables = [var for theory in theory_variables for var in theory]
        return all_variables
