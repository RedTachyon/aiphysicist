from typing import Tuple, Any

import tensorflow as tf
from tensorflow.python import keras


def mlp_model(input_dim: int,
              output_dim: int,
              hidden_sizes: Tuple[int] = (8, 8),
              activation: Any = None,
              output_activation: Any = None):

    layers = [keras.layers.Input((input_dim,))] + [keras.layers.Dense(h, activation) for h in hidden_sizes]
    layers.append(keras.layers.Dense(output_dim, output_activation))

    model = keras.models.Sequential(layers)

    return model
