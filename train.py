from typing import Tuple

import tensorflow as tf
import numpy as np


def read_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv'):
    return np.genfromtxt(path, delimiter=',')


def extract_data(path: str = '../DATA/MYSTERIES/2Dmysteryt50401.csv', T: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    assert T in (1, 2, 3), "T has to be an integer between 1 and 3"
    data = read_data(path)
    X = data[:, :2*T]
    Y = data[:, 2*T:2*T+2]

    return X, Y
