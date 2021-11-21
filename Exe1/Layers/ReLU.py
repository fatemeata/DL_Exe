import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        self.trainable = False
        self._last_input = None

    def forward(self, input_tensor):
        self._last_input = input_tensor
        output = np.maximum(0, input_tensor)
        return output

    def backward(self, error_tensor):
        error_tensor = np.where(self._last_input <= 0, 0, error_tensor)  # if x<=0 -> e(n+1)=0, if x>0 -> e(n+1)= e(n)
        return error_tensor
