import numpy as np

from src.RGrad.tensor import Tensor

def test_tensor():
    tensor1 = Tensor(np.array([1, 2]))
    tensor2 = Tensor(np.array([1, 2]))
    assert tensor2.tensor_index == tensor1.tensor_index + 1