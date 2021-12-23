from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.prob = probability
        self._drop_mask = None

    def forward(self, input_tensor):
        """
        :param input_tensor: input dim: [b:batch_size, n:input_size]
        :return: dropped out input_tensor with given probability
        """

        if self.testing_phase is False:
            # create a random array for removing neurons
            self._drop_mask = np.random.uniform(0, 1, (input_tensor.shape[0], input_tensor.shape[1]))
            self._drop_mask = np.where(self._drop_mask < self.prob, 1, 0)  # if val < prob -> drop_mask = 1, else: 0
            input_tensor = input_tensor * self._drop_mask  # keep neurons with given probability
            input_tensor = input_tensor * (1 / self.prob)
        return input_tensor

    def backward(self, error_tensor):
        """

        :param error_tensor: with dim: [b, m]
        :return: error_tensor which shut down neurons whose got 0 value in forward pass
        """
        error_tensor = self._drop_mask * error_tensor / self.prob
        return error_tensor
