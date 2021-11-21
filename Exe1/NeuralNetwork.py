from copy import deepcopy


class NeuralNetwork:

    def __init__(self, opt):
        self.optimizer = opt
        self.loss = []  # loss value for each iteration after calling train
        self.layers = []  # holds the architecture- based on test: FC- ReLU- FC- SoftMax
        self.data_layer = 0  # provide data and labels
        self.loss_layer = 0  # provide loss and prediction
        self.output = None
        self.y = 0

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.y = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        # input_tensor: output of softmax layer (in this test)

        output_loss = self.loss_layer.forward(input_tensor, label_tensor)
        return output_loss  # scalar value

    def backward(self):

        y = self.loss_layer.backward(self.y)

        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def append_layer(self, layer):
        if layer.trainable:
            opt = deepcopy(self.optimizer)
            layer.optimizer = opt
        self.layers.append(layer)
