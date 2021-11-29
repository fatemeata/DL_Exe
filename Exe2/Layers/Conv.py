from Layers.Base import BaseLayer
# TODO: implement Convolution layer class

class Conv(BaseLayer):
    def __init__(self, str_shape, conv_shape, num_kernels):
        self.stride_shape = str_shape
        self.convolution_shape = conv_shape
        self.num_kernels: num_kernels
        self.trainable = True

    def forward(self, input_tensor):
        """

        :param input_tensor:
        input_tensor- with dim:

        :return:
        output_tensor- providing the tensor for next layer
        dim:
        """

        pass

    def backward(self, error_tensor):
        pass

    def initialize(self, w_init, b_init):
        pass