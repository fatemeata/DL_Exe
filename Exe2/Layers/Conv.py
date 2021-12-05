from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom, Xavier, He, Constant
import numpy as np
from scipy.signal import correlate,convolve
# TODO: implement Convolution layer class


class Conv(BaseLayer):
    def __init__(self, str_shape, conv_shape, num_kernels):
        self.stride_shape = str_shape
        self.convolution_shape = conv_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *conv_shape))
        # print("weight shape: ", self.weights.shape)
        self.bias = np.random.uniform(0, 1, (num_kernels, 1))
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        # weight dimension: [num_kernel, channel, conv_dim, conv_dim]

    def forward(self, input_tensor):
        """

        :param input_tensor
        dim: [b: batch_size, c: channels, y, x (spatial dim)] - [b, c, y, x]

        :return:
        output_tensor- providing the tensor for next layer
        dim: [b, k: number of kernels, y, x]
        """

        if len(input_tensor.shape) == 3:  # input tensor is 1D
            batch, ch, n_x = input_tensor.shape
            f_x = self.convolution_shape[1]
            pad_x = int(f_x / 2)
            # Check whether the kernel size is even or not
            if f_x % 2 == 0:
                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1

            out = np.zeros((batch, self.num_kernels, n_x))
            for i in range(batch):
                x = input_tensor[i, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        x_corr = correlate(x[c, :], self.weights[k, c, :], mode='same')
                        # The basic slice syntax is i:j:k where i is the starting index,
                        # j is the stopping index, and k is the step.
                        out[i, k, :] += x_corr[::self.stride_shape[0]]
                    out[i, k, :] += self.bias[k]
            return out

        else:
            batch, ch, n_x, n_y = input_tensor.shape
            f_x = self.convolution_shape[1]  # conv_filter_width
            f_y = self.convolution_shape[2]  # conv_filter_height

            pad_x = int(f_x / 2)
            pad_y = int(f_y / 2)

            if f_x % 2 == 0:  # if kernel size is even
                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1

            if f_y % 2 == 0:  # if kernel size is even
                n_y = int(((n_y - f_y) + (2 * pad_y) - 1) / self.stride_shape[1]) + 1
            else:
                n_y = int(((n_y - f_y) + (2 * pad_y)) / self.stride_shape[1]) + 1

            out = np.zeros((batch, self.num_kernels, n_x, n_y))  # result
            # input_tensor = np.pad(input_tensor,
            #
            # ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y-1))) # pad the input tensor
            # print("input tensor shape: ", input_tensor.shape)
            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        x_corr = correlate(x[c, :], self.weights[k, c, :], mode='same')
                        # The basic slice syntax is i:j:k where i is the starting index,
                        # j is the stopping index, and k is the step.
                        out[b, k, :] += x_corr[::self.stride_shape[0], ::self.stride_shape[1]]
                    out [b, k, :] += self.bias[k]
            return out

    def backward(self, error_tensor):
        pass

    def initialize(self, w_init, b_init):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = w_init.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = b_init.initialize(self.bias.shape, fan_in, fan_out)

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
