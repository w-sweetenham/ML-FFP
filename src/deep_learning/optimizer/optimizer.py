import numpy as np


class SGD:

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def update(self):
        for param in self.params:
            if param.grad_array is None:
                continue
            param.elems -= param.grad_array * self.lr


class Momentum:

    def __init__(self, params, beta, lr):
        self.params = params
        self.beta = beta
        self.lr = lr
        self.prev_step = []
        for param in params:
            self.prev_step.append(np.zeros(param.shape))

    def update(self):
        for n in range(len(self.params)):
            if self.params[n].grad_array[n] is None:
                continue
            step = self.prev_step[n]*self.beta + (1-self.beta)*self.params[n].grad_array
            self.prev_step[n] = step
            self.params[n].elems -= self.lr*step
