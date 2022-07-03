import numpy as np

from src.deep_learning.RGrad.function import flatten, linear, relu
from src.deep_learning.RGrad.tensor import Tensor


class Param:
    def __init__(self, tensor):
        self.tensor = tensor


class Transform:

    def __init__(self):
        pass

    def params(self):
        params = []
        for att in dir(self):
            if isinstance(getattr(self, att), Param):
                params.append(getattr(self, att).tensor)
            elif isinstance(getattr(self, att), Transform):
                params += getattr(self, att).params()
        return params


class Linear(Transform):

    def __init__(self, input_size, output_size):
        self.weight_param = Param(Tensor(np.random.normal(loc=0, scale=1, size=(output_size, input_size))))

    def __call__(self, inpt):
        return linear(self.weight_param.tensor, inpt)


class ReLUBlock(Transform):

    def __init__(self, input_size, output_size):
        self.linear_transform = Linear(input_size, output_size)
    
    def __call__(self, inpt):
        return relu(self.linear_transform(inpt))


class Flatten(Transform):

    def __init__(self):
        pass
    
    def __call__(self, inpt):
        return flatten(inpt)