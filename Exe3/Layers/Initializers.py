# TODO: implement 4 class for initializing
import numpy as np
import math


class Constant:
    def __init__(self, val=0.1):
        self.constant_val = val

    def initialize(self, weight_shape, fan_in, fan_out):
        # fill the weight matrix with a constant value
        weights = np.full((weight_shape[0], weight_shape[1]), self.constant_val)
        return weights


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weight_shape, fan_in, fan_out):
        weights = np.random.uniform(0, 1, size=weight_shape)
        return weights


class Xavier:

    def __init__(self):
        pass

    # Initializing the weights matrix using Xavier method: N(0, sigma)/ Sigma = sqrt(2/ fan_in + fan_out)
    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = math.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, sigma, size=weight_shape)
        return weights


class He:
    def __init(self):
        pass

    # Initializing the weights matrix using He method: N(0, sigma)/ Sigma = sqrt(2/ fan_in)
    def initialize(self, weight_shape, fan_in, fan_out):
        sigma = math.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, size=weight_shape)

