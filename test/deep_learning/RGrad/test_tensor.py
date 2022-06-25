import numpy as np

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.RGrad.function import matmul, relu, mean


def test_tensor():
    tensor1 = Tensor(np.array([1, 2]))
    tensor2 = Tensor(np.array([1, 2]))
    assert tensor2.tensor_index == tensor1.tensor_index + 1


def test_meta_graph():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([[5, 6], [7, 8]]))
    tensor3 = matmul(tensor1, tensor2)
    tensor4 = relu(tensor2)
    tensor5 = matmul(tensor3, tensor4)
    tensor6 = matmul(tensor5, tensor4)
    tensor7 = relu(tensor5)
    tensor8 = matmul(tensor5, tensor6)
    tensor9 = mean(tensor8)
    meta_graph = tensor9.get_meta_graph()
    assert meta_graph == {tensor1.tensor_index: 1,
                          tensor2.tensor_index: 2,
                          tensor3.tensor_index: 1,
                          tensor4.tensor_index: 2,
                          tensor5.tensor_index: 2,
                          tensor6.tensor_index: 1,
                          tensor8.tensor_index: 1,
                          tensor9.tensor_index: 0}

def test_add_grad_contribution():
    tensor3 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor4 = Tensor(np.array([[5, 6], [7, 8]]))
    tensor2 = matmul(tensor3, tensor4)
    tensor1 = mean(tensor2)
    tensor1.grad_array = np.array(1)
    tensor1.add_grad_contribution(0)
    assert np.allclose(tensor2.grad_array, np.array([[0.25, 0.25], [0.25, 0.25]]))
    tensor2.add_grad_contribution(0)
    assert np.allclose(tensor3.grad_array, np.array([[2.75, 3.75], [2.75, 3.75]]))

def test_backward():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([[5, 6], [7, 8]]))
    tensor3 = matmul(tensor1, tensor2)
    tensor4 = mean(tensor1)
    tensor5 = mean(tensor3)
    tensor5.backward()
    assert tensor4.grad_array is None
    assert np.allclose(tensor5.grad_array, np.array(1))
    assert np.allclose(tensor3.grad_array, np.array([[0.25, 0.25], [0.25, 0.25]]))
    assert np.allclose(tensor1.grad_array, np.array([[2.75, 3.75], [2.75, 3.75]]))
    assert np.allclose(tensor2.grad_array, np.array([[1, 1], [1.5, 1.5]]))

def test_zero_grad():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([[5, 6], [7, 8]]))
    tensor3 = matmul(tensor1, tensor2)
    tensor4 = mean(tensor1)
    tensor5 = mean(tensor3)
    tensor5.backward()
    tensor5.zero_grads()
    assert tensor4.grad_array is None
    assert np.allclose(tensor5.grad_array, np.array(0))
    assert np.allclose(tensor3.grad_array, np.array([[0, 0], [0, 0]]))
    assert np.allclose(tensor1.grad_array, np.array([[0, 0], [0, 0]]))
    assert np.allclose(tensor2.grad_array, np.array([[0, 0], [0, 0]]))
