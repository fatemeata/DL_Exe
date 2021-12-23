import numpy as np
import math


class CrossEntropyLoss:
    def __init__(self):
        self.pred_tensor = 0

    def forward(self, prediction_tensor, label_tensor):
        """

        Args:
            prediction_tensor: np.array- dim: [b:batch_size,n:categories]
            label_tensor: dim: np.array- dim:[b,n]

        Returns:
            loss: scalar float

        """

        self.pred_tensor = prediction_tensor
        # print("prediction tensor: ", self.pred_tensor)
        # print("label tensor: ", label_tensor)
        y_hat = prediction_tensor[np.where(label_tensor == 1)]
        loss = np.sum(-np.log(y_hat + np.finfo(y_hat.dtype).eps), axis=0)
        return loss

    def backward(self, label_tensor):
        return -(label_tensor / self.pred_tensor)

