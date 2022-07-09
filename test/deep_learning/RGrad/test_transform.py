import numpy as np
from src.deep_learning.RGrad.function import mean

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.RGrad.transform import Linear, ReLUBlock


def test_linear():
    linear_transform = Linear(3, 2)
    params = linear_transform.params()
    assert len(params) == 2
    for param in params:
        if param.shape == (2,):
            bias_tensor = param
        elif param.shape == (2, 3):
            weight_tensor = param

    assert weight_tensor.shape == (2, 3)
    assert bias_tensor.shape == (2,)

    inpt_tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    linear_transform.bias_param.tensor.elems[0] = 0.5
    linear_transform.bias_param.tensor.elems[1] = 0.6
    linear_transform.weight_param.tensor.elems[0][0] = 1
    linear_transform.weight_param.tensor.elems[0][1] = 2
    linear_transform.weight_param.tensor.elems[0][2] = 3
    linear_transform.weight_param.tensor.elems[1][0] = 4
    linear_transform.weight_param.tensor.elems[1][1] = 5
    linear_transform.weight_param.tensor.elems[1][2] = 6
    output_tensor = linear_transform(inpt_tensor)
    assert output_tensor.shape == (2, 2)
    assert np.allclose(output_tensor.elems, np.array([[14.5, 32.6], [32.5, 77.6]]))

    scalar_output = mean(output_tensor)
    scalar_output.backward()
    assert weight_tensor.grad_array.shape == (2, 3)
    assert bias_tensor.grad_array.shape == (2,)


def test_relu_block():
    relu_block = ReLUBlock(3, 2)
    params = relu_block.params()
    assert len(params) == 2
    for param in params:
        if param.shape == (2,):
            bias_tensor = param
        elif param.shape == (2, 3):
            weight_tensor = param
    assert weight_tensor.shape == (2, 3)
    assert bias_tensor.shape == (2,)

    inpt_tensor = Tensor(np.ones([4, 3]))
    output_tensor = relu_block(inpt_tensor)
    assert output_tensor.shape == (4, 2)
