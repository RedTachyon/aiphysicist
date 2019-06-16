import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from typing import Any, Optional, Callable, Tuple, List, Dict
import sympy

from utils import get_n_most_frequent

from theory import Theory, Hub
from models import mlp_model
from train import extract_data, iterative_train
from losses import generalized_mean_loss

"""
x_t: [batch, length*dim]
y_t: [batch, dim]
"""

X, Y = extract_data()
M = 4  # Number of theories
M_0 = 2  # Number of theories to be proposed

theory_initializer = lambda: Theory(pred_initializer=mlp_model, domain_initializer=mlp_model,

                                    pred_kwargs={'input_dim': 6,
                                                 'output_dim': 2,
                                                 'hidden_sizes': (8, 8),
                                                 'activation': keras.layers.ReLU(),
                                                 'output_activation': None},

                                    domain_kwargs={'input_dim': 6,
                                                   'output_dim': 1,
                                                   'hidden_sizes': (8, 8),
                                                   'activation': keras.layers.LeakyReLU(0.3),
                                                   'output_activation': None}

                                    )

hub = Hub(M, theory_initializer)

theories = hub.propose_theories(X, Y, M_0)

iterative_train(theories, X, Y, K=101, optimizer_pred=RMSProp(), optimizer_domain=RMSProp())
