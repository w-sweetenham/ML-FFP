"""module containing various optimizers"""
import numpy as np


class SGD:
    """
    class representing stochastic gradient descent optimizer
    """

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def update(self):
        """
        updates the model parameters according to standard gradient descent
        """
        for param in self.params:
            if param.grad_array is None:
                continue
            param.elems -= param.grad_array * self.lr


class Momentum:
    """
    class representing momentum optimizer
    """

    def __init__(self, params, beta, lr):
        self.params = params
        self.beta = beta
        self.lr = lr
        self.prev_step = []
        for param in params:
            self.prev_step.append(np.zeros(param.shape))

    def update(self):
        """
        update parameters according to momentum algorithm update rule
        """
        for index, param in enumerate(self.params):
            if param.grad_array is None:
                continue
            step = (
                self.prev_step[index] * self.beta
                + (1 - self.beta) * param.grad_array
            )
            self.prev_step[index] = step
            param.elems -= self.lr * step
