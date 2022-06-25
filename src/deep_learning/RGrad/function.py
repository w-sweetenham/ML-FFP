from multiprocessing.sharedctypes import Value
import numpy as np

from src.RGrad.tensor import Tensor

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
                derriv_array[index, index] = 1
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
                derriv_array[batch_index][flattened_index][batch_index][image_row][image_col]