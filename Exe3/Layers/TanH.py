from Layers.Base import BaseLayer
import numpy as np


class TanH(BaseLayer):

    def __init__(self):
        super().__init__()
        self.forward_tan = 0

    def forward(self, input_tensor):
        self.forward_tan = np.tanh(input_tensor)
        return self.forward_tan

    def backward(self, error_tensor):
        error_tensor = error_tensor * (1 - self.forward_tan ** 2)
        return error_tensor

