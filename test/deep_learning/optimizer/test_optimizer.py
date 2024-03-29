import numpy as np

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.optimizer.optimizer import SGD, Momentum


def test_sgd():
    tensor = Tensor(np.array([1.0, 1.0]))
    tensor.grad_array = np.array([0.6, 0.2])
    optimizer = SGD([tensor], 0.1)
    optimizer.update()

    assert np.allclose(tensor.elems, np.array([0.94, 0.98]))


def test_momentum():
    tensor = Tensor(np.array([1.0, 1.0]))
    tensor.grad_array = np.array([0.8, 0.4])
    optimizer = Momentum([tensor], 0.8, 0.1)
    optimizer.update()
    
    assert np.allclose(tensor.elems, np.array([0.984, 0.992]))

    tensor.grad_array = np.array([0.5, 0.7])
    optimizer.update()

    assert np.allclose(tensor.elems, np.array([0.9612, 0.9716]))