from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, str_shape, pl_shape):
        self.stride_shape = str_shape
        self.pooling_shape = pl_shape
        self.trainable = False

    def forward(self, input_tensor):
        """

        :param input_tensor: dim: [b:batch, c:channels, x, y]
        :return: output_tensor: dim: [b, c, x/stride, y/stride]
        """

        batch, channel, n_x, n_y = input_tensor.shape
        x_new = int((n_x - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        y_new = int((n_y - self.pooling_shape[1]) / self.stride_shape[1]) + 1
        output = np.zeros((batch, channel, x_new, y_new))

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
        return output

    def backward(self, error_tensor):
        pass