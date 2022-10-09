import numpy as np
from src.deep_learning.RGrad.function import mean

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.RGrad.transform import Conv2D, Linear, ReLUBlock


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


def test_conv2d():
    conv2d_transform = Conv2D(4, 5, 5, 3)
    input_ims = Tensor(np.ones((8, 12, 12, 3)))

    output = conv2d_transform(input_ims)
    assert output.shape == (8, 8, 8, 4)

    output = mean(output)
    output.backward()

    params = conv2d_transform.params()
    assert len(params) == 1
    assert params[0] is conv2d_transform.kernels.tensor
    assert params[0].grad_array.shape == (4, 5, 5, 3)
