import numpy as np
import tensorflow as tf
from typing import Any, Optional, Callable, Tuple, List, Dict
import sympy

from utils import get_n_most_frequent

"""

Design ideas

Sequence dims: [batch, length, dim]
x_t: [batch, length, dim]
y_t: [batch, dim] or [batch, 1, dim] ?



"""


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


class World:
    def __init__(self):
        pass

    def step(self):
        pass

    def observe(self):
        pass


class Theory:
    def __init__(self, pred_initializer: Callable, domain_initializer: Callable,
                 pred_kwargs: Dict, domain_kwargs: Dict):
        """

        Args:
            pred_initializer: function that returns a model, mby neural net
            domain_initializer: as above
        """
        self.predictor = pred_initializer(**pred_kwargs)
        self.classifier = domain_initializer(**domain_kwargs)

    def predict(self, x_t: np.ndarray) -> np.ndarray:
        pass

    def domain(self, x_t: np.ndarray) -> np.ndarray:
        pass

    def evaluate(self, dataset: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        pass


class Hub:
    def __init__(self, initial_theories: int, initializer: Callable[..., Theory]):
        self.theories = [initializer() for _ in range(initial_theories)]

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


if __name__ == '__main__':
    print(tf.__version_)
