import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        BaseLayer.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        # initialize the weights(weights+bias) randomly
        self.weights = np.random.uniform(0, 1, (self.input_size+1, self.output_size))
        self._gradient_weights = None
        self._optimizer = None
        self._last_input = None

    def forward(self, input_tensor):
        """ Forward pass
        Args:
            input_tensor (np.array): Input tensor with dim [b=batch_size, n=input_size]

        Returns:
            output (np.array): Output tensor with dim [b, m=output_size]
        """
        one_vec = np.ones((input_tensor.shape[0], 1))  # dim: [b,1]
        self._last_input = np.concatenate((input_tensor, one_vec), axis=1)
        output = np.dot(self._last_input, self.weights)
        return output

    def backward(self, error_tensor):
        """ Backward pass
        Args:
            error_tensor (np.array): dl/dx with dim [b, m]

        Returns:
            output (np.array): error tensor for previous layer with dim [b, n]

        """
        self._gradient_weights = np.dot(self._last_input.T, error_tensor)  # dim=[n+1, m]
        error_tensor = np.dot(error_tensor, self.weights.T[:, :-1])  # dim = [b, n]
        if self._optimizer: # update weights
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return error_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

# TODO: implement initialize method for fully connected layer

    def initialize(self, w_init, b_init):
        pass
        fan_in = self.input_size
        fan_out = self.output_size
        self.weights[:self.input_size, :] = w_init.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights[self.input_size, :] = b_init.initialize((1, self.output_size), 1, self.output_size)
