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
        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        # print("weight shape: ", self.weights.shape)
        self._gradient_weights = np.random.uniform(0, 1, (self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(0, 1, (num_kernels, 1))
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer = None
        self._bias_optimizer = None
        self._gradient_bias = np.zeros_like(self.bias)
        self._last_input = None
        self._padded_input = None
        self._tmp_array = None
        self.x_corr_shape = 0
        self.pad_X = 0
        self.pad_Y = 0

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
                        x_corr = correlate(x[c], self.weights[k, c], mode='same')
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

            self.pad_X = pad_x
            self.pad_Y = pad_y

            if f_x % 2 == 0:  # if kernel size is even
                n_x = int(((n_x - f_x) + (2 * pad_x) - 1) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x - 1), (pad_y, pad_y))
                                            , mode='constant', constant_values=0.0)
            else:
                n_x = int(((n_x - f_x) + (2 * pad_x)) / self.stride_shape[0]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y))
                                            , mode='constant', constant_values=0.0)

            if f_y % 2 == 0:  # if kernel size is even
                n_y = int(((n_y - f_y) + (2 * pad_y) - 1) / self.stride_shape[1]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y - 1)),
                                            mode='constant', constant_values=0.0)
            else:
                n_y = int(((n_y - f_y) + (2 * pad_y)) / self.stride_shape[1]) + 1
                self._padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y))
                                            , mode='constant', constant_values=0.0)

            out = np.zeros((batch, self.num_kernels, n_x, n_y))  # result

            for b in range(batch):
                x = input_tensor[b, :]
                for k in range(self.num_kernels):
                    for c in range(ch):
                        # x_corr = correlate(self._padded_input[b, c, :], self.weights[k, c, :], mode='valid')
                        x_corr = correlate(x[c], self.weights[k, c], mode='same')  #equals to the above line
                        self.x_corr_shape = x_corr.shape
                        # print("x corr shape: ", x_corr.shape)
                        # self._tmp_array = np.zeros_like(x_corr)
                        # The basic slice syntax is i:j:k where i is the starting index,
                        # j is the stopping index, and k is the step.
                        out[b, k, :] += x_corr[::self.stride_shape[0], ::self.stride_shape[1]]
                    out[b, k, :] += self.bias[k]
            return out

    def backward(self, error_tensor):
        # initialize gradient_weights and gradient_bias !!!
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        # weights: (num_kernel, channel, spatial dimension)
        # rearrange weights: (channel, num_kernel, spatial dimension)

        weights_channel = []
        for i in range(self.convolution_shape[0]):
            weights_kernel = []
            for j in range(self.num_kernels):
                weights_kernel.append(self.weights[j, i])
            weights_channel.append(weights_kernel)
        gradient_layer_kernel = np.array(weights_channel)
        gradient_layer_kernel = np.flip(gradient_layer_kernel, axis=1)  # channel dimension needs to be flipped

        # compute gradients
        batch = []
        for i in range(error_tensor.shape[0]):
            channel = []

            # correct stride
            error_t = np.zeros((error_tensor.shape[1], *self._last_input.shape[2:]))
            if len(self.stride_shape) == 1:  # 1D stride
                error_t[:, ::self.stride_shape[0]] = error_tensor[i]
            else:  # 2D stride
                error_t[:, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[i]

            # convolution along input_channel
            for j in range(gradient_layer_kernel.shape[0]):
                tensor = convolve(error_t, gradient_layer_kernel[j], 'same')
                valid_channel = error_tensor.shape[1] // 2
                tensor = tensor[valid_channel]
                channel.append(tensor)

            # Gradient with respect to the weights

            gradient_kernel = []

            for k in range(self.num_kernels):
                # weights
                # expand (y, x) into (1, y, x) to have same dimension as input_tensor_padding
                temp_tensor = np.expand_dims(error_t[k], axis=0)
                gradient_tensor = correlate(self._padded_input[i, :], temp_tensor, 'valid')
                gradient_kernel.append(gradient_tensor)

                # print("padded input: ", self._padded_input[i, :].shape) (3, 7, 9)
                # print("before expansion: ", error_t[k].shape) (5, 7)
                # print("tmp tensor: ", temp_tensor.shape) (1, 5, 7)

            gradient_weights = np.array(gradient_kernel)
            self.gradient_weights += gradient_weights

            batch.append(channel)

        # Gradient with respect to bias
        if len(self.convolution_shape) == 2:
            # In case we have a batch of 1D signals
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:
            # In case we have a batch of 2D images
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            self._gradient_bias = self._gradient_bias[:, np.newaxis]

        output_tensor = np.array(batch)

        if self._optimizer:  # update weights
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return output_tensor

    # def backward(self, error_tensor):
    #     """
    #     :param error_tensor -> dim: [b: batch, k: number of kernels, y, x: spatial dim]
    #     :return: error_tensor(for the prev layer) -> dim: [b: batch, k: number of kernels, y, x: spatial dim]
    #     """
    #     # weight shape: [k, c, y, x]
    #     # error_tensor: [b, c, y, x]
    #     # gradient_weights:
    #     # gradient_bias:
    #
    #     error_tensor_adapted = np.zeros((error_tensor.shape[0], error_tensor.shape[1], *self._last_input.shape[2:]))
    #
    #     # Write the values of the error tensor at the specific positions based on the used stride parameters
    #     if len(self.convolution_shape) == 2:
    #         # In case we have a batch of 1D signals
    #         error_tensor_adapted[::, ::, ::self.stride_shape[0]] = error_tensor
    #
    #     else:
    #         # In case we have a batch of 2D images
    #         error_tensor_adapted[::, ::, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
    #
    #     if len(self.convolution_shape) == 2:
    #         # In case we have a batch of 1D signals, we pad left and right of our data array
    #         y_pad_left = self.convolution_shape[1] // 2
    #         y_pad_right = self.convolution_shape[1] // 2
    #
    #         # Cut the padded image, so the dimensions of the results still match
    #         if self.convolution_shape[1] % 2 == 0:
    #             y_pad_right -= 1
    #
    #         input_tensor_pad = np.pad(self._last_input, ((0, 0), (0, 0), (y_pad_left, y_pad_right)),
    #                                   mode='constant', constant_values=0.0)
    #
    #     else:
    #         # In case we have a batch of 2D images we pad around the whole image
    #         y_pad_left = self.convolution_shape[1] // 2
    #         y_pad_right = self.convolution_shape[1] // 2
    #         x_pad_left = self.convolution_shape[2] // 2
    #         x_pad_right = self.convolution_shape[2] // 2
    #
    #         # Cut the padded image, so the dimensions of the results still match
    #         if self.convolution_shape[1] % 2 == 0:
    #             y_pad_right -= 1
    #         if self.convolution_shape[2] % 2 == 0:
    #             x_pad_right -= 1
    #
    #     input_tensor_pad = np.pad(self._last_input, ((0, 0), (0, 0), (y_pad_left, y_pad_right),
    #                                                   (x_pad_left, x_pad_right)),
    #                               mode='constant', constant_values=0.0)
    #
    #
    #     channels = self._last_input.shape[1]
    #     batch = self._last_input.shape[0]
    #
    #     # 1d input
    #     if len(error_tensor.shape) == 3:
    #         batch, kernels, x = error_tensor.shape
    #         output = np.zeros((batch, self._last_input.shape[1], self._last_input.shape[2]))
    #         for b in range(batch):
    #             for c in range(channels):
    #                 for k in range(kernels):
    #                     error_t = np.zeros(output[b, c, :].shape)
    #                     error_t[::self.stride_shape[0]] = error_tensor[b, k, :]
    #                     output[b, c, :] += convolve(error_t, self.weights[k, c, :], mode='same')
    #                     self._gradient_weights[k, c, :] += correlate(self._padded_input[b, c, :], error_t, mode='valid')
    #             for k in range(kernels):
    #                 self._gradient_bias[k] += np.sum(error_tensor[b, k, :])
    #     else:
    #         output = np.zeros_like(self._last_input)
    #
    #         if len(self.convolution_shape) == 2:
    #             # In case we have a batch of 1D signals
    #             self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
    #
    #         else:
    #             # In case we have a batch of 2D images
    #             self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
    #             self._gradient_bias = self._gradient_bias[:, np.newaxis]
    #
    #         for channel in range(error_tensor.shape[1]):
    #             kernel = 0
    #
    #             for input_image_pad, error_image_adopted in zip(input_tensor_pad, error_tensor_adapted[:, channel]):
    #                 # Correlate the padded input images with the adpted error image channel-wise
    #                 # Increase the dimension of error_image_adopted to match the dimensions of input_image_pad
    #                 kernel += correlate(input_image_pad, np.array([error_image_adopted]), mode='valid')
    #
    #             # Add the calculated kernel to the gradient weights for the respective channel.
    #             if self.gradient_weights is None:
    #                 # Transform first kernel to a batch representation, to be able to stack all kernels together
    #                 self.gradient_weights = np.array([kernel])
    #
    #             else:
    #                 # Add the new generated kernel to the stack of kernels
    #                 self.gradient_weights = np.concatenate((self.gradient_weights, [kernel]))
    #
    #             # Cut the gradient weights onto the same shape as the weights are, so the optimization works.
    #         self.gradient_weights = self.gradient_weights[:self.weights.shape[0]]
    #
    #         for b in range(batch):
    #             for c in range(channels):
    #                 for k in range(self.num_kernels):
    #                     error_t = np.zeros(output[b, c].shape)
    #                     error_t[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k]
    #                     output[b, c] += convolve(error_t, self.weights[k, c, :], mode='same')
    #
    #                     # self._gradient_weights[k, c] += correlate(self._padded_input[b, c], error_t, mode='valid')
    #
    #         if self._optimizer:  # update weights
    #             self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
    #             self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
    #
    #     return output

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