import numpy as np
import math

class Sgd:
    def __init__(self, lr):
        self.learning_rate = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        return weight_tensor
