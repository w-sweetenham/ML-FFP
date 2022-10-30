import numpy as np

from src.deep_learning.RGrad.tensor import Tensor


class MatmulFunction:

    @staticmethod
    def forward(a, b):
        return np.matmul(a.elems, b.elems)

    @staticmethod
    def backward(a, b, return_index):
        result_size = (a.shape[0], b.shape[1])
        if return_index == 0:
            for output_index in np.ndindex(result_size):
                derriv_array = np.zeros(a.shape)
                output_row, output_col = output_index
                derriv_array[output_row] = b.elems[:, output_col]
                yield output_index, derriv_array
        elif return_index == 1:
            for output_index in np.ndindex(result_size):
                output_row, output_col = output_index
                derriv_array = np.zeros(b.shape)
                derriv_array[:, output_col] = a.elems[output_row]
                yield output_index, derriv_array
        else:
            raise ValueError(f'invalid return index: {return_index}')

    @staticmethod
    def has_valid_backward(index):
        if index in {0, 1}:
            return True
        else:
            return False


def matmul(a, b):
    return Tensor(MatmulFunction.forward(a, b), (a, b), MatmulFunction)


class ReLUFunction:

    @staticmethod
    def forward(a):
        return a.elems * (a.elems > 0)

    @staticmethod
    def backward(a, index):
        if index != 0:
            raise ValueError('invalid index specified')
        for output_index in np.ndindex(a.shape):
            derriv_array = np.zeros(a.shape)
            if a.elems[output_index] >= 0.0:
                derriv_array[output_index] = 1.0
            yield output_index, derriv_array

    @staticmethod
    def has_valid_backward(index):
        return index == 0


def relu(a):
    return Tensor(ReLUFunction.forward(a), (a,), ReLUFunction)


class MeanFunction:

    @staticmethod
    def forward(a):
        return np.mean(a.elems)

    @staticmethod
    def backward(a, index):
        if index != 0:
            raise ValueError('invalid index specified')
        yield None, np.ones(a.shape)/a.elems.size
    
    @staticmethod
    def has_valid_backward(index):
        return index == 0


def mean(a):
    return Tensor(MeanFunction.forward(a), (a,), MeanFunction)


class CrossEntropyFunction:

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
            yield None, derriv_array
        elif index == 1:
            return None
        else:
            raise ValueError(f'invalid index passed to backwards method: {index}')

    @staticmethod
    def has_valid_backward(index):
        return index == 0


def cross_entropy(logits, labels):
    return Tensor(CrossEntropyFunction.forward(logits, labels), (logits, labels), CrossEntropyFunction)


class LinearFunction:

    @staticmethod
    def forward(weight_tensor, vector_tensor):
        return np.matmul(vector_tensor.elems, np.transpose(weight_tensor.elems))

    @staticmethod
    def backward(weight_tensor, vector_tensor, index):
        output_shape = (vector_tensor.shape[0], weight_tensor.shape[0])
        if index == 0:
            for output_index in np.ndindex(output_shape):
                output_row, output_col = output_index
                derriv_array = np.zeros(weight_tensor.shape)
                derriv_array[output_col] = vector_tensor.elems[output_row]
                yield output_index, derriv_array
        elif index == 1:
            for output_index in np.ndindex(output_shape):
                output_row, output_col = output_index
                derriv_array = np.zeros(vector_tensor.shape)
                derriv_array[output_row] = weight_tensor.elems[output_col]
                yield output_index, derriv_array
        else:
            raise ValueError(f'invalid index: {index}')

    @staticmethod
    def has_valid_backward(index):
        return index in {0, 1}


def linear(weight_tensor, vector_tensor):
    return Tensor(LinearFunction.forward(weight_tensor, vector_tensor), (weight_tensor, vector_tensor), LinearFunction)


class Flatten:

    @staticmethod
    def forward(image_tensor):
        return np.reshape(image_tensor.elems, (image_tensor.shape[0], image_tensor.shape[1]*image_tensor.shape[2]))

    @staticmethod
    def backward(image_tensor, index):
        if index != 0:
            raise ValueError(f'invalid index: {index}')
        output_shape = (image_tensor.shape[0], image_tensor.shape[1]*image_tensor.shape[2])
        for output_index in np.ndindex(output_shape):
            derriv_array = np.zeros(image_tensor.shape)
            output_row, output_col = output_index
            image_row = output_col // image_tensor.shape[2]
            image_col = output_col % image_tensor.shape[2]
            derriv_array[output_row][image_row][image_col] = 1
            yield output_index, derriv_array

    @staticmethod
    def has_valid_backward(index):
        return index == 0


def flatten(tensor):
    return Tensor(Flatten.forward(tensor), (tensor,), Flatten)


class Add:

    @staticmethod
    def forward(tensor1, tensor2):
        new_array = np.copy(tensor1.elems)
        for batch_num, array in enumerate(new_array):
            new_array[batch_num] += tensor2.elems
        return new_array

    @staticmethod
    def backward(tensor1, tensor2, index):
        if index == 0:
            for output_index in np.ndindex(tensor1.shape):
                derriv_array = np.zeros(tensor1.shape)
                derriv_array[output_index] = 1
                yield output_index, derriv_array
        elif index == 1:
            for output_index in np.ndindex(tensor1.shape):
                derriv_array = np.zeros(tensor2.shape)
                derriv_array[output_index[1:]] = 1
                yield output_index, derriv_array
        else:
            raise ValueError(f'invalid index: {index}')

    @staticmethod
    def has_valid_backward(index):
        return index in {0 ,1}


def add(tensor1, tensor2):
    return Tensor(Add.forward(tensor1, tensor2), (tensor1, tensor2), Add)


class Conv2d:

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
            for output_index in np.ndindex(output_size):
                derriv_array = np.zeros(images_tensor.shape)
                batch_num, output_row, output_col, kernel_num = output_index
                for im_row in range(output_row, output_row + num_kernel_rows):
                    for im_col in range(output_col, output_col + num_kernel_cols):
                        for im_depth in range(depth):
                            image_pos = (batch_num, im_row, im_col, im_depth)
                            derriv_array[image_pos] = kernels_tensor.elems[kernel_num][im_row-output_row][im_col-output_col][im_depth]
                yield output_index, derriv_array
        elif index == 1:
            for output_index in np.ndindex(output_size):
                derriv_array = np.zeros(kernels_tensor.shape)
                batch_num, output_row, output_col, kernel_num = output_index
                for kernel_row in range(num_kernel_rows):
                    for kernel_col in range(num_kernel_cols):
                        for kernel_depth in range(depth):
                            kernel_pos = (kernel_num, kernel_row, kernel_col, kernel_depth)
                            derriv_array[kernel_pos] = images_tensor.elems[batch_num][kernel_row+output_row][kernel_col+output_col][kernel_depth]
                yield output_index, derriv_array

    @staticmethod
    def has_valid_backward(index):
        return index in {0 ,1}


def conv2d(images_tensor, kernels_tensor):
    return Tensor(Conv2d.forward(images_tensor, kernels_tensor), (images_tensor, kernels_tensor), Conv2d)


class AddDimension:

    @staticmethod
    def forward(tensor):
        return np.expand_dims(np.copy(tensor.elems), len(tensor.shape))

    @staticmethod
    def backward(tensor, index):
        if index != 0:
            raise ValueError(f'invalid index: {index}')

        for output_index in np.ndindex(tensor.shape + (1,)):
            derriv_array = np.zeros(tensor.shape)
            derriv_array[output_index[:-1]] = 1
            yield output_index, derriv_array

    @staticmethod
    def has_valid_backward(index):
        return index == 0


def add_dimension(tensor):
    return Tensor(AddDimension.forward(tensor), (tensor,), AddDimension)


class Pad:

    @staticmethod
    def forward(input_tensor, pad_size_tensor):
        new_array = np.zeros((input_tensor.shape[0], input_tensor.shape[1]+(2*pad_size_tensor.elems), input_tensor.shape[2]+(2*pad_size_tensor.elems)))
        new_array[:, pad_size_tensor.elems:input_tensor.shape[1]+pad_size_tensor.elems, pad_size_tensor.elems:input_tensor.shape[2]+pad_size_tensor.elems] = input_tensor.elems
        return new_array

    @staticmethod
    def backward(input_tensor, pad_size_tensor, index):
        if index == 0:
            output_shape = (input_tensor.shape[0], input_tensor.shape[1]+(2*pad_size_tensor.elems), input_tensor.shape[2])
            for output_index in np.ndindex(output_shape):
                derriv_array = np.zeros_like(input_tensor.elems)
                if (output_index[1] < pad_size_tensor.elems or
                    output_index[1] >= input_tensor.shape[1] + pad_size_tensor.elems or
                    output_index[2] < pad_size_tensor.elems or
                    output_index[2] >= input_tensor.shape[2] + pad_size_tensor.elems):

                    yield output_index, derriv_array
                else:
                    derriv_array[output_index[0]][output_index[1]-pad_size_tensor.elems][output_index[2]-pad_size_tensor.elems] = 1
                    yield output_index, derriv_array
        else:
            raise ValueError(f'invalid index for backward pass: {index}')

    @staticmethod
    def has_valid_backward(index):
        return index == 0

    
def pad(input_tensor, pad_tensor):
    return Tensor(Pad.forward(input_tensor, pad_tensor), (input_tensor, pad_tensor), Pad)
