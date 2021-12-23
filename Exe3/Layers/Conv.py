from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom, Xavier, He, Constant
import numpy as np
from scipy.signal import correlate, convolve
import copy

# TODO: implement Convolution layer class

# A convolution layer transforms an input volume into an output volume of different size.

class Conv(BaseLayer):
    def __init__(self, str_shape, conv_shape, num_kernels):
        self.stride_shape = str_shape
        self.convolution_shape = conv_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, (num_kernels, 1))
        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_weights = 0
        self._gradient_bias = 0
        self._last_input = None
        self._padded_input = None

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

        if len(input_tensor.shape) == 3:  # input tensor is 1D and therefore Filter is 1D [b, c, x]
            batch, ch, n_x = input_tensor.shape

            f_x = self.convolution_shape[1]  # [c, m]
            pad_x = int(f_x / 2) # half of the kernel size

            # Check whether the kernel size is even or not
            if f_x % 2 == 0:  # if kernel size is even

                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x - 1)))
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x)))

            out = np.zeros((batch, self.num_kernels, n_x))  #output shape

            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        x_corr = correlate(x[c], self.weights[k, c], mode='same')
                        out[b, k, :] += x_corr[::self.stride_shape[0]]
                    out[b, k, :] += self.bias[k]
            return out

        else:
            batch, ch, n_x, n_y = input_tensor.shape
            f_x = self.convolution_shape[1]  # conv_filter_width
            f_y = self.convolution_shape[2]  # conv_filter_height

            pad_x_1 = int(f_x / 2)
            pad_y_1 = int(f_y / 2)

            if f_x % 2 == 0:
                n_x = int(((n_x - f_x) + (2 * pad_x_1) - 1) / self.stride_shape[0]) + 1
                pad_x_2 = pad_x_1 - 1
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x_1)) / self.stride_shape[0]) + 1
                pad_x_2 = pad_x_1

            if f_y % 2 == 0:
                n_y = int(((n_y - f_y) + (2 * pad_y_1) - 1) / self.stride_shape[1]) + 1
                pad_y_2 = pad_y_1 - 1

            else:
                n_y = int(((n_y - f_y) + (2 * pad_y_1)) / self.stride_shape[1]) + 1
                pad_y_2 = pad_y_1

            self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x_1, pad_x_2), (pad_y_1, pad_y_2))
                                        , mode='constant', constant_values=0.0)

            out = np.zeros((batch, self.num_kernels, n_x, n_y))  # output shape

            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        # x_corr = correlate(self._padded_input[b, c, :], self.weights[k, c, :], mode='valid')
                        x_corr = correlate(x[c], self.weights[k, c], mode='same')  # equals to the above line
                        out[b, k, :] += x_corr[::self.stride_shape[0], ::self.stride_shape[1]]  # down sampling
                    out[b, k, :] += self.bias[k]
            return out



    def backward(self, error_tensor):
        """

        :param error_tensor: [b, k, x, y]
        :return: dim: [b, c, x, y]
        """

        # initialize gradient_weights and gradient_bias
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        # weights: (num_kernel, channel, spatial dimension)
        # rearrange weights: (channel, num_kernel, spatial dimension)

        batch = error_tensor.shape[0]

        weight_ch_list = []

        for i in range(self.convolution_shape[0]):
            weights_kernel_list = []
            for j in range(self.num_kernels):
                weights_kernel_list.append(self.weights[j, i])
            weight_ch_list.append(weights_kernel_list)

        gradient_layer_k = np.array(weight_ch_list)
        gradient_layer_k = np.flip(gradient_layer_k, axis=1)  # rearrange the dimension

        output_list = []
        for i in range(batch):  # iterating over batch
            channel_list = []

            # correct stride / Up sampling
            error_t = np.zeros((error_tensor.shape[1], *self._last_input.shape[2:]))
            if len(self.stride_shape) == 1:  # 1D stride
                error_t[:, ::self.stride_shape[0]] = error_tensor[i]
            else:  # 2D stride
                error_t[:, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]

            for j in range(gradient_layer_k.shape[0]):
                tensor_2d = convolve(error_t, gradient_layer_k[j], 'same')
                ch = error_tensor.shape[1] // 2
                tensor_2d = tensor_2d[ch]
                channel_list.append(tensor_2d)

            # Gradient with respect to the weights
            kernel_gradient = []

            for k in range(self.num_kernels):
                # weights
                # expand (x, y) into (1, x, y) to have same dimension as padded input
                error_tensor_expanded = np.expand_dims(error_t[k], axis=0)
                gradient_tensor = correlate(self._padded_input[i, :], error_tensor_expanded, 'valid')
                kernel_gradient.append(gradient_tensor)

                # print("padded input: ", self._padded_input[i, :].shape) (3, 7, 9)
                # print("before expansion: ", error_t[k].shape) (5, 7)
                # print("tmp tensor: ", temp_tensor.shape) (1, 5, 7)

            weights_gradient = np.array(kernel_gradient)
            self.gradient_weights += weights_gradient

            output_list.append(channel_list)

        # Gradient with respect to bias
        if len(self.convolution_shape) == 2:
            # 1D input
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:
            # 2D input
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))  # sum over the batch and spatial dimension
            self._gradient_bias = self._gradient_bias[:, np.newaxis]

        # update weights
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        output = np.array(output_list)
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
    def gradient_bias(self, val):
        self._gradient_bias = val