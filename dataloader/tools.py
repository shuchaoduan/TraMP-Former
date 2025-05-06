
import random
import numpy as np
import torch
from scipy.interpolate import CubicSpline      # for warping
from math import sin, cos


class _Reverse_seq(object):
    def __init__(self):
        pass
    def __call__(self, x):
        if np.random.rand(1)[0] > .5:
            reversed_x = x[::-1]  # np array
            return reversed_x
        else:
            return x

class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """
    def __init__(self, augment_list, k):
        self.augment_list = augment_list
        self.k = k

    def __call__(self, x):
        # ops = random.choices(self.augment_list, k=self.k)
        ops = self.augment_list
        for aug in ops:
            x = aug(x)
        return x
