from re import A
import numpy as np

from src.RGrad.tensor import Tensor
from src.RGrad.function import Matmul, matmul

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