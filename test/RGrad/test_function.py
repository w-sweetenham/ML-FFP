from re import A
import numpy as np

from src.RGrad.tensor import Tensor
from src.RGrad.function import Matmul, matmul, ReLU, relu, Mean, mean


def test_matmul_forward():
    tensor_a = Tensor(np.array([[1, 2], [3, 4]], dtype=float))
    tensor_b = Tensor(np.array([[5, 6], [7, 8]], dtype=float))
    tensor_c = matmul(tensor_a, tensor_b)

    assert np.allclose(tensor_c.elems, np.array([[19, 22], [43, 50]], dtype=float))
    assert len(tensor_c.parents) == 2
    assert tensor_c.parents[0] is tensor_a
    assert tensor_c.parents[1] is tensor_b


def test_matmul_backward():
    tensor_a = Tensor(np.array([[1, 2], [3, 4]], dtype=float))
    tensor_b = Tensor(np.array([[5, 6], [7, 8]], dtype=float))

    a_derriv_array = Matmul.backward(tensor_a, tensor_b, 0)
    assert np.allclose(a_derriv_array[0][0], np.array([[5, 7], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[0][1], np.array([[6, 8], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[1][0], np.array([[0, 0], [5, 7]], dtype=float))
    assert np.allclose(a_derriv_array[1][1], np.array([[0, 0], [6, 8]], dtype=float))

    b_derriv_array = Matmul.backward(tensor_a, tensor_b, 1)
    assert np.allclose(a_derriv_array[0][0], np.array([[5, 7], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[0][1], np.array([[6, 8], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[1][0], np.array([[0, 0], [5, 7]], dtype=float))
    assert np.allclose(a_derriv_array[1][1], np.array([[0, 0], [6, 8]], dtype=float))


def test_relu_forward():
    input_tensor = Tensor(np.array([1, -1, 0.5]))
    output_tensor = relu(input_tensor)
    np.allclose(output_tensor.elems, np.array([1, 0, 0.5]))
    assert len(output_tensor.parents) == 1
    assert output_tensor.parents[0] is input_tensor
    assert output_tensor.function == ReLU


def test_relu_backward():
    input_tensor = Tensor(np.array([1.5, 0, -1]))
    derriv_array = ReLU.backward(input_tensor, 0)
    assert np.allclose(derriv_array, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))


def test_mean_forward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]], dtype=float))
    output_tensor = mean(input_tensor)
    assert np.isclose(output_tensor.elems, 2.5)
    assert len(output_tensor.parents) == 1
    assert output_tensor.parents[0] is input_tensor


def test_mean_backward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    derriv_array = Mean.backward(input_tensor, 0)
    assert np.allclose(derriv_array, np.array([[0.25, 0.25], [0.25, 0.25]]))