import numpy as np
from src.RGrad.function import mean

from src.RGrad.tensor import Tensor
from src.RGrad.transform import Linear, ReLUBlock


def test_linear():
    linear_transform = Linear(3, 2)
    params = linear_transform.params()
    assert len(params) == 1
    weight_tensor = params[0]
    assert weight_tensor.shape == (2, 3)

    inpt_tensor = Tensor(np.ones([4, 3]))
    output_tensor = linear_transform(inpt_tensor)
    assert output_tensor.shape == (4, 2)
    assert output_tensor.parents == (weight_tensor, inpt_tensor)

    scalar_output = mean(output_tensor)
    scalar_output.backward()
    assert weight_tensor.grad_array.shape == (2, 3)


def test_relu_block():
    relu_block = ReLUBlock(3, 2)
    params = relu_block.params()
    assert len(params) == 1
    weight_tensor = params[0]
    assert weight_tensor.shape == (2, 3)

    inpt_tensor = Tensor(np.ones([4, 3]))
    output_tensor = relu_block(inpt_tensor)
    assert output_tensor.shape == (4, 2)
