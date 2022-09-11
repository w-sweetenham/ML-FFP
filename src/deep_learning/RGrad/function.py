from multiprocessing.sharedctypes import Value
from tkinter import image_names
import numpy as np

from src.deep_learning.RGrad.tensor import Tensor

class MatmulFunction:

    def __init__(self):
        pass

    @staticmethod
    def forward(a, b):
        return np.matmul(a.elems, b.elems)

    @staticmethod
    def backward(a, b, return_index):
        result_size = [a.shape[0], b.shape[1]]
        if return_index == 0:
            a_derriv_array = np.zeros(result_size + [a.shape[0], a.shape[1]])
            for i in range(result_size[0]):
                for j in range(result_size[1]):
                    for y in range(a.shape[1]):
                        a_derriv_array[i][j][i][y] = b.elems[y][j]
            return a_derriv_array
        elif return_index == 1:
            b_derriv_array = np.zeros(result_size + [b.shape[0], b.shape[1]])
            for i in range(result_size[0]):
                for j in range(result_size[1]):
                    for x in range(b.shape[0]):
                        b_derriv_array[i][j][x][j] = a.elems[i][x]
            return b_derriv_array
        return


def matmul(a, b):
    return Tensor(MatmulFunction.forward(a, b), (a, b), MatmulFunction)


class ReLUFunction:

    def __init__(self):
        pass

    @staticmethod
    def forward(a):
        return a.elems * (a.elems > 0)

    @staticmethod
    def backward(a, index):
        if index != 0:
            raise ValueError('invalid index specified')
        result_size = list(a.shape)
        derriv_array = np.zeros(result_size + result_size)
        for index in np.ndindex(a.shape):
            if a.elems[index] > 0:
                derriv_array[index][index] = 1
        return derriv_array


def relu(a):
    return Tensor(ReLUFunction.forward(a), (a,), ReLUFunction)


class MeanFunction:

    def __init__(self):
        pass

    @staticmethod
    def forward(a):
        return np.mean(a.elems)

    @staticmethod
    def backward(a, index):
        if index != 0:
            raise ValueError('invalid index specified')
        return np.ones(a.shape)/a.elems.size


def mean(a):
    return Tensor(MeanFunction.forward(a), (a,), MeanFunction)


class CrossEntropyFunction:

    def __init__(self):
        pass

    @staticmethod
    def forward(logits, labels):
        B = logits.shape[0]
        loss = 0
        for i in range(B):
            label = labels.elems[i]
            vec = logits.elems[i, :]
            loss -= vec[label] - np.log(sum(np.exp(vec - max(vec)))) - max(vec)
        return loss/B

    @staticmethod
    def backward(logits, labels, index):
        B = logits.shape[0]
        if index == 0:
            derriv_array = np.exp(logits.elems)
            derriv_array = np.divide(derriv_array, np.reshape(np.sum(derriv_array, 1), [len(derriv_array), 1]))
            for i in range(len(labels.elems)):
                derriv_array[i][labels.elems[i]] -= 1
            derriv_array /= B
            return derriv_array
        elif index == 1:
            return None
        else:
            raise ValueError(f'invalid index passed to backwards method: {index}')


def cross_entropy(logits, labels):
    return Tensor(CrossEntropyFunction.forward(logits, labels), (logits, labels), CrossEntropyFunction)


class LinearFunction:

    def __init__(self):
        pass

    @staticmethod
    def forward(weight_tensor, vector_tensor):
        return np.matmul(vector_tensor.elems, np.transpose(weight_tensor.elems))

    @staticmethod
    def backward(weight_tensor, vector_tensor, index):
        vector_tensor_transposed = Tensor(np.transpose(vector_tensor.elems))
        if index == 0:
            return np.transpose(MatmulFunction.backward(weight_tensor, vector_tensor_transposed, 0), [1, 0, 2, 3])
        elif index == 1:
            return np.transpose(MatmulFunction.backward(weight_tensor, vector_tensor_transposed, 1), [1, 0, 3, 2])
        else:
            raise ValueError(f'invalid index: {index}')


def linear(weight_tensor, vector_tensor):
    return Tensor(LinearFunction.forward(weight_tensor, vector_tensor), (weight_tensor, vector_tensor), LinearFunction)


class Flatten:

    def __init__(self):
        pass

    @staticmethod
    def forward(image_tensor):
        return np.reshape(image_tensor.elems, (image_tensor.shape[0], image_tensor.shape[1]*image_tensor.shape[2]))

    @staticmethod
    def backward(image_tensor, index):
        if index != 0:
            raise ValueError(f'invalid index: {index}')
        derriv_array = np.zeros([image_tensor.shape[0], image_tensor.shape[1]*image_tensor.shape[2]] + list(image_tensor.shape))
        for batch_index in range(image_tensor.shape[0]):
            for flattened_index in range(image_tensor.shape[1]*image_tensor.shape[2]):
                image_row = flattened_index // image_tensor.shape[2]
                image_col = flattened_index % image_tensor.shape[2]
                derriv_array[batch_index][flattened_index][batch_index][image_row][image_col] = 1.0
        return derriv_array


def flatten(tensor):
    return Tensor(Flatten.forward(tensor), (tensor,), Flatten)


class Add:

    def __init__(self):
        pass

    @staticmethod
    def forward(tensor1, tensor2):
        new_array = np.copy(tensor1.elems)
        for n in range(len(new_array)):
            for index in np.ndindex(new_array.shape[1:]):
                new_array[n][index] += tensor2.elems[index]
        return new_array

    @staticmethod
    def backward(tensor1, tensor2, index):
        if index == 0:
            derriv_array = np.zeros(tensor1.shape + tensor1.shape)
            for array_index in np.ndindex(tensor1.shape):
                derriv_array[array_index][array_index] = 1
            return derriv_array
        elif index == 1:
            derriv_array = np.zeros(tensor1.shape + tensor2.shape)
            for array_index in np.ndindex(tensor1.shape):
                tensor2_index = array_index[1:]
                derriv_array[array_index][tensor2_index] = 1
            return derriv_array
        else:
            raise ValueError(f'invalid index: {index}')


def add(tensor1, tensor2):
    return Tensor(Add.forward(tensor1, tensor2), (tensor1, tensor2), Add)


class Conv2d:

    def __init__(self):
        pass

    @staticmethod
    def forward(images_tensor, kernels_tensor):
        batch_size = images_tensor.shape[0]
        num_kernels = kernels_tensor.shape[0]
        depth = images_tensor.shape[3]
        num_image_rows = images_tensor.shape[1]
        num_image_cols = images_tensor.shape[2]
        num_kernel_rows = kernels_tensor.shape[1]
        num_kernel_cols = kernels_tensor.shape[2]
        if depth != kernels_tensor.shape[3]:
            raise ValueError(f'image depth doesn\'t match kernel depth: {depth} vs {kernels_tensor.shape[3]}')
        output_array = np.zeros((batch_size, num_image_rows-num_kernel_rows+1, num_image_cols-num_kernel_cols+1, num_kernels))
        for batch_index in range(batch_size):
            for output_row_index in range(num_image_rows-num_kernel_rows+1):
                for output_col_index in range(num_image_cols-num_kernel_cols+1):
                    for kernel_index in range(num_kernels):
                        elem_val = 0
                        for image_row_index in range(output_row_index, output_row_index+num_kernel_rows):
                            kernel_row_index = image_row_index - output_row_index
                            for image_col_index in range(output_col_index, output_col_index+num_kernel_cols):
                                kernel_col_index = image_col_index - output_col_index
                                for depth_index in range(depth):
                                    elem_val += images_tensor.elems[batch_index][image_row_index][image_col_index][depth_index]*kernels_tensor.elems[kernel_index][kernel_row_index][kernel_col_index][depth_index]
                        output_array[batch_index][output_row_index][output_col_index][kernel_index] = elem_val
        return output_array

    @staticmethod
    def backward(images_tensor, kernels_tensor, index):
        batch_size = images_tensor.shape[0]
        num_kernels = kernels_tensor.shape[0]
        depth = images_tensor.shape[3]
        num_image_rows = images_tensor.shape[1]
        num_image_cols = images_tensor.shape[2]
        num_kernel_rows = kernels_tensor.shape[1]
        num_kernel_cols = kernels_tensor.shape[2]
        output_size = (batch_size, num_image_rows-num_kernel_rows+1, num_image_cols-num_kernel_cols+1, num_kernels)
        if index == 0:
            derriv_array = np.zeros(output_size + images_tensor.shape)
            for batch_num, output_row, output_col, kernel_num in np.ndindex(output_size):
                output_pos = (batch_num, output_row, output_col, kernel_num)
                for kernel_row, kernel_col, im_depth in np.ndindex((num_kernel_rows, num_kernel_cols, depth)):
                    image_pos = (batch_num, output_row+kernel_row, output_col+kernel_col, im_depth)
                    derriv_array[output_pos][image_pos] = kernels_tensor.elems[kernel_num][kernel_row][kernel_col][im_depth]


