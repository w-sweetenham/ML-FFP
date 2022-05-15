from multiprocessing.sharedctypes import Value
import numpy as np

from src.RGrad.tensor import Tensor

class Matmul:

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
    return Tensor(Matmul.forward(a, b), (a, b), Matmul)


class ReLU:

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
    return Tensor(ReLU.forward(a), (a,), ReLU)


class Mean:

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
    return Tensor(Mean.forward(a), (a,), Mean)


class CrossEntropy:

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
            print(derriv_array)
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
    return Tensor(CrossEntropy.forward(logits, labels), (logits, labels), CrossEntropy)


class Linear:

    def __init__(self):
        pass

    @staticmethod
    def forward(weight_tensor, vector_tensor):
        return np.matmul(vector_tensor.elems, np.transpose(weight_tensor.elems))

    @staticmethod
    def backward(weight_tensor, vector_tensor, index):
        vector_tensor_transposed = Tensor(np.transpose(vector_tensor.elems))
        if index == 0:
            return np.transpose(Matmul.backward(weight_tensor, vector_tensor_transposed, 0), [1, 0, 2, 3])
        elif index == 1:
            return np.transpose(Matmul.backward(weight_tensor, vector_tensor_transposed, 1), [1, 0, 3, 2])
        else:
            raise ValueError(f'invalid index: {index}')


def linear(weight_tensor, vector_tensor):
    return Tensor(Linear.forward(weight_tensor, vector_tensor), (weight_tensor, vector_tensor), Linear)
