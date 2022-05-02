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