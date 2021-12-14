import numpy as np
import math

class Sgd:
    def __init__(self, lr):
        self.learning_rate = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        return weight_tensor


# TODO: implement sgd with momentum optimizer
class SgdWithMomentum:
    def __init__(self, lr, mr):
        self.learning_rate = lr
        self.momentum_rate = mr
        self.momentum_term = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.momentum_term = (self.momentum_rate * self.momentum_term) - (self.learning_rate * gradient_tensor)
        weight_tensor = weight_tensor + self.momentum_term
        return weight_tensor


# TODO: implement ADAM optimizer
class Adam:
    def __init__(self, lr, mu, rho):
        self.learning_rate = lr
        self.mu = mu
        self.rho = rho
        self.v_term = 0
        self.r_term = 0
        self.it = 1
        # self.epsilon = np.finfo(np.float).eps
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.v_term = (self.mu * self.v_term) + ((1 - self.mu) * gradient_tensor)
        self.r_term = self.rho * self.r_term + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        v_hat = self.v_term / (1 - math.pow(self.mu, self.it))
        r_hat = self.r_term / (1 - math.pow(self.rho, self.it))

        adam_term = v_hat / (np.sqrt(r_hat) + self.epsilon)

        weight_tensor = weight_tensor - self.learning_rate * adam_term
        self.it = self.it + 1

        # print("weight tensor: ", weight_tensor)

        return weight_tensor
