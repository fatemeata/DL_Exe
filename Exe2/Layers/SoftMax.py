import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):

    def __init__(self):
        self.trainable = False
        self._output = 0

    def forward(self, input_tensor):

        """

        Args:
            input_tensor: dim: [b:batch_size, n:categories]

        Returns:
            self.output: np.array- dim: [b,n]

        """

        input_tensor -= input_tensor.max(axis=1, keepdims=True)
        exp = np.exp(input_tensor)
        self._output = exp / np.sum(exp, axis=1, keepdims=True)
        return self._output


    def backward(self, error_tensor):
        """

        Args:
            error_tensor: dim: [b, n]

        Returns:
            error_tensor: dim: [b,n]

        """

        summation = np.sum(np.multiply(error_tensor, self._output), axis=1, keepdims=True)
        error_tensor -= summation
        error_tensor = np.multiply(self._output, error_tensor)
        return error_tensor

