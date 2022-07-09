import numpy as np

from src.deep_learning.RGrad.function import flatten, linear, relu, add
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

    def __init__(self, input_size, output_size, initialization='xavier'):
        if initialization == 'xavier':
            init_weights = np.random.uniform(-1/(input_size**0.5), 1/(input_size**0.5), size=(output_size, input_size))
        elif initialization == 'normalized_xavier':
            init_weights = np.random.uniform(-((6**0.5)/(input_size + output_size)**0.5), ((6**0.5)/(input_size + output_size)**0.5), size=(output_size, input_size))
        elif initialization  == 'he':
            init_weights = np.random.normal(loc=0, scale=2/(input_size**0.5), size=(output_size, input_size))
        else:
            raise ValueError(f'invalid initialization specified: {initialization}')
        self.weight_param = Param(Tensor(init_weights))
        self.bias_param = Param(Tensor(np.zeros(output_size)))

    def __call__(self, inpt):
        return add(linear(self.weight_param.tensor, inpt), self.bias_param.tensor)


class ReLUBlock(Transform):

    def __init__(self, input_size, output_size):
        self.linear_transform = Linear(input_size, output_size, initialization='he')
    
    def __call__(self, inpt):
        return relu(self.linear_transform(inpt))


class Flatten(Transform):

    def __init__(self):
        pass
    
    def __call__(self, inpt):
        return flatten(inpt)