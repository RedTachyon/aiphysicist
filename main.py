import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from typing import Any, Optional, Callable, Tuple, List, Dict
import sympy

from utils import get_n_most_frequent

from theory import Theory, Hub
from models import mlp_model

"""
x_t: [batch, length*dim]
y_t: [batch, dim]
"""

if __name__ == '__main__':
    theory_initializer = lambda: Theory(pred_initializer=mlp_model, domain_initializer=mlp_model,

                                        pred_kwargs={'input_dim': 6,
                                                     'output_dim': 2,
                                                     'hidden_sizes': (8, 8),
                                                     'activation': None,
                                                     'output_activation': None},

                                        domain_kwargs={'input_dim': 6,
                                                       'output_dim': 1,
                                                       'hidden_sizes': (8, 8),
                                                       'activation': keras.layers.LeakyReLU(0.3),
                                                       'output_activation': None}

                                        )

    hub = Hub(5, theory_initializer)
