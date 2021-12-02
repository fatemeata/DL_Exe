from Layers.Base import BaseLayer
import numpy as np
# TODO: implement the flatten class


class Flatten(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.input_tensor_shape = 0

    def forward(self, input_tensor):
        """
        :param input_tensor: dim= (b: batch_size, input_shape[0], input_shape[1], input_shape[2])
        :return: input_tensor for next layer: dim= (b, the product of input_shape)

        """
        self.input_tensor_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(input_tensor.shape[0], np.prod(input_tensor.shape[1:]))
        return input_tensor

    def backward(self, error_tensor):
        """

        :param error_tensor: dim= [b, the product of input_shape]
        :return: error_tensor for the previous layer: dim=(b: batch_size, input_shape[0],
        input_shape[1], input_shape[2])
        """
        error_tensor = error_tensor.reshape(self.input_tensor_shape[0],
                                            self.input_tensor_shape[1],
                                            self.input_tensor_shape[2],
                                            self.input_tensor_shape[3])

        return error_tensor
