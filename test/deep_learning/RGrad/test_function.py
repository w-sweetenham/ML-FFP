import numpy as np

from src.deep_learning.RGrad.tensor import Tensor
from src.deep_learning.RGrad.function import MatmulFunction, matmul, ReLUFunction, relu, MeanFunction, mean, cross_entropy, CrossEntropyFunction, LinearFunction, Flatten, Add, Conv2d


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

    a_derriv_array = MatmulFunction.backward(tensor_a, tensor_b, 0)
    assert np.allclose(a_derriv_array[0][0], np.array([[5, 7], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[0][1], np.array([[6, 8], [0, 0]], dtype=float))
    assert np.allclose(a_derriv_array[1][0], np.array([[0, 0], [5, 7]], dtype=float))
    assert np.allclose(a_derriv_array[1][1], np.array([[0, 0], [6, 8]], dtype=float))

    b_derriv_array = MatmulFunction.backward(tensor_a, tensor_b, 1)
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
    assert output_tensor.function == ReLUFunction


def test_relu_backward():
    input_tensor = Tensor(np.array([1.5, 0, -1]))
    derriv_array = ReLUFunction.backward(input_tensor, 0)
    assert np.allclose(derriv_array, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))


def test_mean_forward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]], dtype=float))
    output_tensor = mean(input_tensor)
    assert np.isclose(output_tensor.elems, 2.5)
    assert len(output_tensor.parents) == 1
    assert output_tensor.parents[0] is input_tensor


def test_mean_backward():
    input_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    derriv_array = MeanFunction.backward(input_tensor, 0)
    assert np.allclose(derriv_array, np.array([[0.25, 0.25], [0.25, 0.25]]))


def test_cross_entropy_forward():
    logits = Tensor(np.array([[1, 2], [5, 3], [3, 6]]))
    labels = Tensor(np.array([1, 0, 0]))
    loss = cross_entropy(logits, labels)
    assert np.isclose(1.16292556, loss.elems)


def test_cross_entropy_backward():
    logits = Tensor(np.array([[4, 3], [5, 6], [10, 5]]))
    labels = Tensor(np.array([0, 0, 1]))
    loss = CrossEntropyFunction.backward(logits, labels, 0)
    assert np.allclose(np.array([[-0.0896471406, 0.08964714046], [-0.2436861929, 0.2436861929], [0.331102383, -0.331102383]]), loss)


def test_linear_forward():
    weight_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    vec_tensor = Tensor(np.array([[5, 7], [6, 8]]))
    output = LinearFunction.forward(weight_tensor, vec_tensor)
    assert np.allclose(output, np.array([[19, 43], [22, 50]]))


def test_linear_backward():
    weight_tensor = Tensor(np.array([[1, 2], [3, 4]]))
    vec_tensor = Tensor(np.array([[5, 7], [6, 8]]))
    weight_derriv = LinearFunction.backward(weight_tensor, vec_tensor, 0)
    vec_derriv = LinearFunction.backward(weight_tensor, vec_tensor, 1)
    correct_weight_derriv = np.array([[[[5, 7], [0, 0]], [[0, 0], [5, 7]]], [[[6, 8], [0, 0]], [[0, 0], [6, 8]]]])
    correct_vec_derriv = np.array([[[[1, 2], [0, 0]], [[3, 4], [0, 0]]], [[[0, 0], [1, 2]], [[0, 0], [3, 4]]]])
    assert np.allclose(weight_derriv, correct_weight_derriv)
    assert np.allclose(vec_derriv, correct_vec_derriv)


def test_flatten_forward():
    image_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    flattened_array = Flatten.forward(image_tensor)
    assert np.all(flattened_array == np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))


def test_flatten_backward():
    image_tensor = Tensor(np.array([[[1, 2], [4, 5]], [[7, 8], [11, 12]]]))
    flattened_image = Flatten.forward(image_tensor)
    grad_array = Flatten.backward(image_tensor, 0)
    non_zero_positions = [(0, 0, 0, 0, 0), (0, 1, 0, 0, 1), (0, 2, 0, 1, 0), (0, 3, 0, 1, 1),
                          (1, 0, 1, 0, 0), (1, 1, 1, 0, 1), (1, 2, 1, 1, 0), (1, 3, 1, 1, 1)]
    assert grad_array.shape == (2, 4, 2, 2, 2)
    for index in np.ndindex(grad_array.shape):
        if index in non_zero_positions:
            assert np.isclose(grad_array[index], 1.0)
        else:
            assert np.isclose(grad_array[index], 0.0)


def test_add_forward():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([5, 6]))
    tensor3_elems = Add.forward(tensor1, tensor2)
    assert np.all(tensor3_elems == np.array([[6, 8], [8, 10]]))


def test_add_backward():
    tensor1 = Tensor(np.array([[1, 2], [3, 4]]))
    tensor2 = Tensor(np.array([5, 6]))
    backward_array0 = Add.backward(tensor1, tensor2, 0)
    backward_array1 = Add.backward(tensor1, tensor2, 1)
    derriv_array0_correct = np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, 1]]]])
    derriv_array1_correct = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    assert np.all(backward_array0 == derriv_array0_correct)
    assert np.all(backward_array1 == derriv_array1_correct)


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
    
    image_derriv_tensor = Conv2d.backward(images_tensor, kernel_tensor, 0)
    assert image_derriv_tensor[0][0][0][1][0][0][0][0] == 9
    assert image_derriv_tensor[0][0][0][1][0][1][0][0] == 13
    assert image_derriv_tensor[0][0][0][1][0][0][1][1] == 12
    
    kernel_derriv_tensor = Conv2d.backward(images_tensor, kernel_tensor, 1)
    assert kernel_derriv_tensor[1][0][1][0][0][0][0][0] == 15
    assert kernel_derriv_tensor[1][0][1][0][0][1][1][1] == 24
