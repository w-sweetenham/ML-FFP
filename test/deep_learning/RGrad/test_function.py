import numpy as np

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.RGrad.function import (
    AddDimension, 
    MatmulFunction, 
    add_dimension, 
    matmul, 
    ReLUFunction, 
    relu, 
    MeanFunction, 
    mean, 
    cross_entropy, 
    CrossEntropyFunction, 
    LinearFunction, 
    Flatten, 
    Add, 
    Conv2d, 
    Pad,
    Sigmoid
)


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

    derriv_arrays = []
    for output_index, derriv_array in MatmulFunction.backward(tensor_a, tensor_b, 0):
        derriv_arrays.append(derriv_array)
    assert np.allclose(derriv_arrays[0], np.array([[5, 7], [0, 0]], dtype=float))
    assert np.allclose(derriv_arrays[1], np.array([[6, 8], [0, 0]], dtype=float))
    assert np.allclose(derriv_arrays[2], np.array([[0, 0], [5, 7]], dtype=float))
    assert np.allclose(derriv_arrays[3], np.array([[0, 0], [6, 8]], dtype=float))

    derriv_arrays = []
    for output_index, derriv_array in MatmulFunction.backward(tensor_a, tensor_b, 1):
        derriv_arrays.append(derriv_array)
    assert np.allclose(derriv_arrays[0], np.array([[1, 0], [2, 0]], dtype=float))
    assert np.allclose(derriv_arrays[1], np.array([[0, 1], [0, 2]], dtype=float))
    assert np.allclose(derriv_arrays[2], np.array([[3, 0], [4, 0]], dtype=float))
    assert np.allclose(derriv_arrays[3], np.array([[0, 3], [0, 4]], dtype=float))


def test_relu_forward():
    input_tensor = Tensor(np.array([1, -1, 0.5]))
    output_tensor = relu(input_tensor)
    np.allclose(output_tensor.elems, np.array([1, 0, 0.5]))
    assert len(output_tensor.parents) == 1
    assert output_tensor.parents[0] is input_tensor
    assert output_tensor.function == ReLUFunction


def test_relu_backward():
    input_tensor = Tensor(np.array([1.5, 0.01, -1]))
    derriv_arrays = []
    for index, derriv_array in ReLUFunction.backward(input_tensor, 0):
        derriv_arrays.append(derriv_array)
    assert np.allclose(derriv_arrays[0], np.array([1, 0, 0]))
    assert np.allclose(derriv_arrays[1], np.array([0, 1, 0]))
    assert np.allclose(derriv_arrays[2], np.array([0, 0, 0]))


def test_mean_forward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]], dtype=float))
    output_tensor = mean(input_tensor)
    assert np.isclose(output_tensor.elems, 2.5)
    assert len(output_tensor.parents) == 1
    assert output_tensor.parents[0] is input_tensor


def test_mean_backward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    for index, derriv_array in MeanFunction.backward(input_tensor, 0):
        assert np.allclose(derriv_array, np.array([[0.25, 0.25], [0.25, 0.25]]))


def test_cross_entropy_forward():
    logits = Tensor(np.array([[1, 2], [5, 3], [3, 6]]))
    labels = Tensor(np.array([1, 0, 0]))
    loss = cross_entropy(logits, labels)
    assert np.isclose(1.16292556, loss.elems)


def test_cross_entropy_backward():
    logits = Tensor(np.array([[4, 3], [5, 6], [10, 5]]))
    labels = Tensor(np.array([0, 0, 1]))
    derriv_arrays = []
    for _, derriv_array in CrossEntropyFunction.backward(logits, labels, 0):
        derriv_arrays.append(derriv_array)
    assert len(derriv_arrays) == 1
    assert np.allclose(np.array([[-0.0896471406, 0.08964714046], [-0.2436861929, 0.2436861929], [0.331102383, -0.331102383]]), derriv_arrays[0])


def test_linear_forward():
    weight_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    vec_tensor = Tensor(np.array([[5, 7], [6, 8]]))
    output = LinearFunction.forward(weight_tensor, vec_tensor)
    assert np.allclose(output, np.array([[19, 43], [22, 50]]))


def test_linear_backward():
    weight_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    vec_tensor = Tensor(np.array([[5, 7], [6, 8]]))
    
    weight_derriv_arrays = []
    for index, derriv_array in LinearFunction.backward(weight_tensor, vec_tensor, 0):
        weight_derriv_arrays.append(derriv_array)
    
    vec_derriv_arrays = []
    for index, derriv_array in LinearFunction.backward(weight_tensor, vec_tensor, 1):
        vec_derriv_arrays.append(derriv_array)
    
    assert np.allclose(weight_derriv_arrays[0], [[5, 7], [0, 0]])
    assert np.allclose(weight_derriv_arrays[1], [[0, 0], [5, 7]])
    assert np.allclose(weight_derriv_arrays[2], [[6, 8], [0, 0]])
    assert np.allclose(weight_derriv_arrays[3], [[0, 0], [6, 8]])

    assert np.allclose(vec_derriv_arrays[0], [[1, 2], [0, 0]])
    assert np.allclose(vec_derriv_arrays[1], [[3, 4], [0, 0]])
    assert np.allclose(vec_derriv_arrays[2], [[0, 0], [1, 2]])
    assert np.allclose(vec_derriv_arrays[3], [[0, 0], [3, 4]])


def test_flatten_forward():
    image_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    flattened_array = Flatten.forward(image_tensor)
    assert np.all(flattened_array == np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))


def test_flatten_backward():
    image_tensor = Tensor(np.array([[[1, 2], [4, 5]], [[7, 8], [11, 12]]]))
    flattened_image = Flatten.forward(image_tensor)
    derriv_arrays = []
    for index, derriv_array in Flatten.backward(image_tensor, 0):
        derriv_arrays.append(derriv_array)
    non_zero_positions = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                          (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert len(derriv_arrays) == 8
    for index, non_zero_coords in enumerate(non_zero_positions):
        for derriv_array_coords in np.ndindex(derriv_arrays[index].shape):
            if derriv_array_coords == non_zero_coords:
                assert np.isclose(derriv_arrays[index][derriv_array_coords], 1.0)
            else:
                assert np.isclose(derriv_arrays[index][derriv_array_coords], 0.0)


def test_add_forward():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([5, 6]))
    tensor3_elems = Add.forward(tensor1, tensor2)
    assert np.all(tensor3_elems == np.array([[6, 8], [8, 10]]))


def test_add_backward():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([5, 6]))
    derriv_arrays = []
    for index, derriv_array in Add.backward(tensor1, tensor2, 0):
        derriv_arrays.append(derriv_array)

    assert np.allclose(derriv_arrays[0], np.array([[1, 0], [0, 0]]))
    assert np.allclose(derriv_arrays[1], np.array([[0, 1], [0, 0]]))
    assert np.allclose(derriv_arrays[2], np.array([[0, 0], [1, 0]]))
    assert np.allclose(derriv_arrays[3], np.array([[0, 0], [0, 1]]))

    derriv_arrays = []
    for index, derriv_array in Add.backward(tensor1, tensor2, 1):
        derriv_arrays.append(derriv_array)

    assert np.allclose(derriv_arrays[0], np.array([1, 0]))
    assert np.allclose(derriv_arrays[1], np.array([0, 1]))
    assert np.allclose(derriv_arrays[2], np.array([1, 0]))
    assert np.allclose(derriv_arrays[3], np.array([0, 1]))


def test_conv2d_forward():
    images_tensor = Tensor(np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]]))
    kernel_tensor = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]], [[[17, 18], [19, 20]], [[21, 22], [23, 24]]]]))
    output_tensor = Conv2d.forward(images_tensor, kernel_tensor)
    assert output_tensor.shape == (2, 1, 2, 3)
    assert np.all(output_tensor[0][0][0] == np.array([256, 608, 960]))
    assert np.all(output_tensor[1][0][1] == np.array([760, 2008, 3256]))


def test_conv2d_backward():
    images_tensor = Tensor(np.array([[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], [[[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]]]))
    kernel_tensor = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]], [[[17, 18], [19, 20]], [[21, 22], [23, 24]]]]))
    
    derriv_arrays = []
    for index, derriv_array in Conv2d.backward(images_tensor, kernel_tensor, 0):
        derriv_arrays.append(derriv_array)

    assert derriv_arrays[1][0][0][0][0] == 9
    assert derriv_arrays[1][0][1][0][0] == 13
    assert derriv_arrays[1][0][0][1][1] == 12

    derriv_arrays = []
    indexes = []
    for index, derriv_array in Conv2d.backward(images_tensor, kernel_tensor, 1):
        derriv_arrays.append(derriv_array)
        indexes.append(index)

    assert derriv_arrays[9][0][0][0][0] == 15
    assert derriv_arrays[9][0][1][1][1] == 24


def test_add_dimension_forward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    output_tensor = add_dimension(input_tensor)
    assert output_tensor.shape == (2, 2, 1)
    assert np.all(output_tensor.elems == np.array([[[1], [2]], [[3], [4]]]))


def test_add_dimension_backward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    derriv_tensors = []
    for index, derriv_tensor in AddDimension.backward(input_tensor, 0):
        derriv_tensors.append(derriv_tensor)

    assert len(derriv_tensors) == 4
    assert np.allclose(derriv_tensors[0], np.array([[1, 0], [0, 0]]))
    assert np.allclose(derriv_tensors[1], np.array([[0, 1], [0, 0]]))
    assert np.allclose(derriv_tensors[2], np.array([[0, 0], [1, 0]]))
    assert np.allclose(derriv_tensors[3], np.array([[0, 0], [0, 1]]))


def test_pad_forward():
    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[6, 7, 8], [9, 10, 11]]]))
    pad_tensor = Tensor(np.array(1))

    correct_padded_tensor = np.zeros((2, 4, 5))
    correct_padded_tensor[0][1] = np.array([0, 1, 2, 3, 0])
    correct_padded_tensor[0][2] = np.array([0, 4, 5, 6, 0])
    correct_padded_tensor[1][1] = np.array([0, 6, 7, 8, 0])
    correct_padded_tensor[1][2] = np.array([0, 9, 10, 11, 0])
    assert np.all(Pad.forward(input_tensor, pad_tensor) == correct_padded_tensor)


def test_pad_backward():
    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[6, 7, 8], [9, 10, 11]]]))
    pad_tensor = Tensor(np.array(1))

    for output_index, derriv_array in Pad.backward(input_tensor, pad_tensor, 0):
        correct_derriv_array = np.zeros((2, 2, 3))
        if output_index[1] in {0, 3} or output_index[2] in {0, 4}:
            assert np.all(derriv_array == correct_derriv_array)
        else:
            correct_derriv_array[output_index[0]][output_index[1]-pad_tensor.elems][output_index[2]-pad_tensor.elems] = 1
            assert np.all(derriv_array == correct_derriv_array)


def test_sigmoid_forward():
    input_tensor = Tensor(np.array([1, 2]))
    output_array = Sigmoid.forward(input_tensor)
    assert np.allclose(output_array, np.array([0.7310585786, 0.880797078]))

def test_sigmoid_backward():
    input_tensor = Tensor(np.array([1, 2]))
    derriv_tensors = []
    for output_index, derriv_tensor in Sigmoid.backward(input_tensor, 0):
        derriv_tensors.append(derriv_tensor)
    assert np.allclose(derriv_tensors[0], np.array([0.1966119332, 0]))
    assert np.allclose(derriv_tensors[1], np.array([0, 0.1049935854]))