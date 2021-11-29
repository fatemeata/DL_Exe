# TODO: implement 4 class for initializing
import numpy as np
import math

class Constant:
    def __init__(self, val=0.1):
        self.constant_val = val

    def initialize(self, weight_shape, fan_in, fan_out):
        weights = np.full((fan_in, fan_out), self.constant_val)
        return weights


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weight_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, (weight_shape[0], weight_shape[1]))
        return weights


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = math.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, sigma, (weight_shape[0], weight_shape[1]))
        return weights


class He:
    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = math.sqrt(2 / fan_in)
        weights = np.random.normal(0, sigma, (weight_shape[0], weight_shape[1]))
        return weights
