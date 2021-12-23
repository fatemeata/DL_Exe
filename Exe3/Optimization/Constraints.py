import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self.regular_weight = alpha


    def calculate_gradient(self, weights):
        weights = self.regular_weight * np.sign(weights)
        return weights

    def norm(self, weights):
        # l1_norm_term : alpha * ||w||
        l1_norm_term = self.regular_weight * np.sum(np.abs(weights))
        return l1_norm_term


class L2_Regularizer:
    def __init__(self, alpha):
        self.regular_weight = alpha

    def calculate_gradient(self, weights):
        weights = self.regular_weight * weights
        return weights

    def norm(self, weights):
        # l2_norm_term : alpha * ||w^2||2
        l2_norm_term = self.regular_weight * np.sum(weights ** 2)
        return l2_norm_term

