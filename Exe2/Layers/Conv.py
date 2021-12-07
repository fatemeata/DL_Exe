from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom, Xavier, He, Constant
import numpy as np
from scipy.signal import correlate, convolve
import copy

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
        self._bias_optimizer = None
        self._gradient_weights = 0
        self._gradient_bias = 0
        self._last_input = None
        self._padded_input = None
        self._tmp_array = None

        # weight dimension: [num_kernel, channel, conv_dim, conv_dim]

    def forward(self, input_tensor):
        """

        :param input_tensor
        dim: [b: batch_size, c: channels, y, x (spatial dim)] - [b, c, y, x]

        :return:
        output_tensor- providing the tensor for next layer
        dim: [b, k: number of kernels, y, x]
        """
        self._last_input = input_tensor
        if len(input_tensor.shape) == 3:  # input tensor is 1D
            batch, ch, n_x = input_tensor.shape
            f_x = self.convolution_shape[1]
            pad_x = int(f_x / 2)
            # Check whether the kernel size is even or not
            if f_x % 2 == 0:  # if kernel size is even
                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x - 1)))
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x)))

            out = np.zeros((batch, self.num_kernels, n_x))
            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        x_corr = correlate(x[c, :], self.weights[k, c, :], mode='same')
                        print("x correlation shape:", x_corr.shape)
                        # The basic slice syntax is i:j:k where i is the starting index,
                        # j is the stopping index, and k is the step.
                        out[b, k, :] += x_corr[::self.stride_shape[0]]
                    out[b, k, :] += self.bias[k]
            return out

        else:
            batch, ch, n_x, n_y = input_tensor.shape
            f_x = self.convolution_shape[1]  # conv_filter_width
            f_y = self.convolution_shape[2]  # conv_filter_height

            pad_x = int(f_x / 2)
            pad_y = int(f_y / 2)

            if f_x % 2 == 0:  # if kernel size is even
                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x - 1), (pad_y, pad_y)))
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

            if f_y % 2 == 0:  # if kernel size is even
                n_y = int(((n_y - f_y) + (2 * pad_y) - 1) / self.stride_shape[1]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y - 1)))
            else:
                n_y = int(((n_y - f_y) + (2 * pad_y)) / self.stride_shape[1]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

            out = np.zeros((batch, self.num_kernels, n_x, n_y))  # result

            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        x_corr = correlate(self._padded_input[b, c, :], self.weights[k, c, :], mode='valid')
                        # x_2_corr = correlate(x[c, :], self.weights[k, c, :], mode='same')  #equals to the above line
                        # print("x corr shape: ", x_corr.shape)
                        # self._tmp_array = np.zeros_like(x_corr)
                        # The basic slice syntax is i:j:k where i is the starting index,
                        # j is the stopping index, and k is the step.
                        out[b, k, :] += x_corr[::self.stride_shape[0], ::self.stride_shape[1]]
                    out[b, k, :] += self.bias[k]
            return out

    def backward(self, error_tensor):
        """
        :param error_tensor -> dim: [b: batch, k: number of kernels, y, x: spatial dim]
        :return: error_tensor(for the prev layer) -> dim: [b: batch, k: number of kernels, y, x: spatial dim]
        """
        # weight shape: [k, c, y, x]
        # error_tensor: [b, c, y, x]
        # gradient_weights:
        # gradient_bias:

        channels = self._last_input.shape[1]
        self._gradient_bias = np.full_like(self.bias, 0)
        self._gradient_weights = np.full_like(self.weights, 0)
        # 1d input
        if len(error_tensor.shape) == 3:
            batch, kernels, x = error_tensor.shape
            output = np.zeros((batch, self._last_input.shape[1], self._last_input.shape[2]))
            for b in range(batch):
                for c in range(channels):
                    for k in range(kernels):
                        error_t = np.zeros(output[b, c, :].shape)
                        error_t[::self.stride_shape[0]] = error_tensor[b, k, :]
                        output[b, c, :] += convolve(error_t, self.weights[k, c, :], mode='same')
        else:
            batch, kernels, x, y = error_tensor.shape
            output = np.zeros((batch,  *self._last_input.shape[1:]))

            for b in range(batch):
                for c in range(channels):
                    for k in range(kernels):
                        error_t = np.zeros(output[b, c, :].shape)
                        error_t[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k, :]
                        output[b, c, :] += convolve(error_t, self.weights[k, c, :], mode='same')

        return output

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
        self._bias_optimizer = copy.deepcopy(val)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_weights(self, val):
        self._gradient_bias = val
