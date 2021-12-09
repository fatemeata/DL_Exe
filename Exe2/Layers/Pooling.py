from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, str_shape, pl_shape):
        self.stride_shape = str_shape
        self.pooling_shape = pl_shape
        self.trainable = False
        self.max_locations = None
        self.input_shape = None

    def forward(self, input_tensor):
        """

        :param input_tensor: dim: [b:batch, c:channels, x, y]
        :return: output_tensor: dim: [b, c, x/stride, y/stride]
        """

        batch, channel, n_x, n_y = input_tensor.shape
        self.last_input = input_tensor

        x_new = int((n_x - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        y_new = int((n_y - self.pooling_shape[1]) / self.stride_shape[1]) + 1
        output = np.zeros((batch, channel, x_new, y_new))
        self.max_locations = np.zeros_like(input_tensor)
        for b in range(batch):
            input_b = input_tensor[b]
            for i in range(x_new):
                x_start = i * self.stride_shape[0]
                x_stop = x_start + self.pooling_shape[0]
                for j in range(y_new):
                    y_start = j * self.stride_shape[1]
                    y_stop = y_start + self.pooling_shape[1]
                    for c in range(channel):
                        win = input_b[c, x_start:x_stop, y_start:y_stop]
                        output[b, c, i, j] += np.max(win)
                        # create an array where the max is 1 and others are 0 for every windows
                        mask = (win == np.max(win)).astype(int)
                        self.max_locations[b, c, x_start:x_stop, y_start:y_stop] += mask
        return output

    def backward(self, error_tensor):
        """

        :param error_tensor: [b, c, x, y]
        :return: [b, c, x_new, y_new]
        """
        # we only need dl/dx
        batch, channel, n_x, n_y = error_tensor.shape
        batch, channel, n_x_new, n_y_new = self.last_input.shape
        output = np.zeros_like(self.last_input)
        for b in range(batch):
            for c in range(channel):
                for i in range(n_x):
                    for j in range(n_y):
                        x_start = i * self.stride_shape[0]
                        x_stop = x_start + self.pooling_shape[0]
                        y_start = j * self.stride_shape[1]
                        y_stop = y_start + self.pooling_shape[1]

                        win = self.last_input[b, c, x_start:x_stop, y_start:y_stop]
                        mask = win == np.max(win)
                        output[b, c, x_start:x_stop, y_start:y_stop] += np.multiply(mask, error_tensor[b, c, i, j])

        return output
