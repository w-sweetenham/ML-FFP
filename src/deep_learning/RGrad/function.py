"""module defining various mathematical operations on tensors. Each mathematical
    operation defined has a class and a python function. The class defines 3
    static methods: a forward method, a backward method and an index checking
    method. The forward method accepts tensor objects of the input tensors and
    returns a numpy array of the result of the operation applied to those
    tensors. The backward method returns tensors of the partial derrivatives of
    the output of the operation with respect to the input tensors. It takes as
    input the same arguments as the forward method but in addition takes in a
    return_index argument specifying which of the input tensors to take the
    derrivates with respect to. It returns a generator object which iterates
    through each position in the output tensor, returning the index of the
    position and the tensor of partial derrivatives of that output element with
    respect to the specified input. The third method checks whether a given
    return index is valid i.e. whether it makes sense to return the derrivative
    of the output wrt that element."""
import numpy as np

from src.deep_learning.RGrad.tensor import Tensor


class MatmulFunction:
    """
    class representing the matrix multiplication function
    """
    @staticmethod
    def forward(a, b):
        """
        forward pass of the matrix multiplication operation.

        Args:
            a (numpy array): first matrix
            b (numpy array): second matrix

        Returns:
            numpy array: result of matrix multiplication of a and b
        """
        return np.matmul(a.elems, b.elems)

    @staticmethod
    def backward(a, b, return_index):
        """
        backward pass of the matrix multiplication operation

        Args:
            a (numpy array): first input matrix
            b (numpy array): second input matrix
            return_index (int): 0 if desiring derrivative wrt a or 1 if
                desiring derrivative wrt b

        Raises:
            ValueError: if return index is not 0 or 1

        Yields:
            tuple: tuple of numpy arrays of the index in the output tensor and
                derrivative of the corresponding output element wrt the
                specified input tensor
        """
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
        """
        valid return index checking method for matrix multiplication 
        """
        return index in {0, 1}


def matmul(a, b):
    """
    python function corresponding to the matrix multiplication operation

    Args:
        a (Tensor): tensor of first matrix
        b (Tensor): tensor of second matrix

    Returns:
        Tensor: tensor result of multiplication
    """
    return Tensor(MatmulFunction.forward(a, b), (a, b), MatmulFunction)


class ReLUFunction:
    """
    class corresponding to the RELU operation
    """

    @staticmethod
    def forward(a):
        """
        forward pass of the RELU operation

        Args:
            a (Tensor): tensor to apply operation to

        Returns:
            numpy array: numpy array of the resulting tensor
        """
        return a.elems * (a.elems > 0)

    @staticmethod
    def backward(a, index):
        """
        backward pass of the RELU operation

        Args:
            a (Tensor): tensor of the input to the RELU oepration
            index (int): should be 0

        Raises:
            ValueError: if index other than 0 given

        Yields:
            _type_: _description_
        """
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
            if len(image_tensor.shape) == 4 and image_tensor.shape[3] == 1:
                derriv_array = np.expand_dims(derriv_array, axis=3)
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


class Sigmoid:

    @staticmethod
    def forward(input_tensor):
        return 1/(1+np.exp(-1*input_tensor.elems))

    @staticmethod
    def backward(input_tensor, index):
        if index == 0:
            for output_index in np.ndindex(input_tensor.shape):
                derriv_array = np.zeros_like(input_tensor.elems, dtype=float)
                exponent = np.exp(-1*input_tensor.elems[output_index])
                derriv_array[output_index] = exponent/(1+exponent)**2
                yield output_index, derriv_array
        else:
            raise ValueError(f'invalid index: {index}')

    @staticmethod
    def has_valid_backward(index):
        return index == 0


def sigmoid(input_tensor):
    return Tensor(Sigmoid.forward(input_tensor), (input_tensor,), Sigmoid)


class MaxPool:
    @staticmethod
    def forward(input_tensor, window_size_tensor):
        num_rows = input_tensor.shape[1]
        num_cols = input_tensor.shape[2]

        window_row_indexes = [(n*window_size_tensor.elems, (n+1)*window_size_tensor.elems) for n in range(num_rows // window_size_tensor.elems)]
        if window_row_indexes[-1][1] < num_rows:
            window_row_indexes.append(((num_rows // window_size_tensor.elems)*window_size_tensor.elems, num_rows))

        window_col_indexes = [(n*window_size_tensor.elems, (n+1)*window_size_tensor.elems) for n in range(num_cols // window_size_tensor.elems)]
        if window_col_indexes[-1][1] < num_cols:
            window_col_indexes.append(((num_cols // window_size_tensor.elems)*window_size_tensor.elems, num_cols))

        output_array = np.empty((input_tensor.shape[0], len(window_row_indexes), len(window_col_indexes), input_tensor.shape[3]), dtype=np.float32)
        for output_row_index, input_row_indexes in enumerate(window_row_indexes):
            for output_col_index, input_col_indexes in enumerate(window_col_indexes):
                start_row, end_row = input_row_indexes
                start_col, end_col = input_col_indexes
                output_array[:, output_row_index, output_col_index] = np.max(input_tensor.elems[:, start_row:end_row, start_col:end_col], axis=(1,2))
        
        return output_array
    
    @staticmethod
    def backward(input_tensor, window_size_tensor, index):
        if index != 0:
            raise ValueError(f'invalid return index: {index}')
        num_batches, num_rows, num_cols, num_channels = input_tensor.shape

        window_row_indexes = [(n*window_size_tensor.elems, (n+1)*window_size_tensor.elems) for n in range(num_rows // window_size_tensor.elems)]
        if window_row_indexes[-1][1] < num_rows:
            window_row_indexes.append(((num_rows // window_size_tensor.elems)*window_size_tensor.elems, num_rows))

        window_col_indexes = [(n*window_size_tensor.elems, (n+1)*window_size_tensor.elems) for n in range(num_cols // window_size_tensor.elems)]
        if window_col_indexes[-1][1] < num_cols:
            window_col_indexes.append(((num_cols // window_size_tensor.elems)*window_size_tensor.elems, num_cols))

        for batch_index in range(num_batches):
            for output_row, input_row_indexes in enumerate(window_row_indexes):
                for output_col, input_col_indexes in enumerate(window_col_indexes):
                    for channel in range(num_channels):
                        start_row, end_row = input_row_indexes
                        start_col, end_col = input_col_indexes
                        input_slice = input_tensor.elems[batch_index, start_row:end_row, start_col:end_col, channel]
                        row_size = end_col - start_col
                        max_index_flat = np.argmax(input_slice)
                        row_max_index = max_index_flat // row_size
                        col_max_index = max_index_flat % row_size
                        return_array = np.zeros(input_tensor.shape, dtype=np.float32)
                        return_array[batch_index, start_row + row_max_index, start_col + col_max_index, channel] = 1.0
                        output_index = (batch_index, output_row, output_col, channel)
                        yield output_index, return_array
        
    @staticmethod
    def has_valid_backward(index):
        return index == 0
    

def max_pool(input_tensor, window_size_tensor):
    return Tensor(MaxPool.forward(input_tensor, window_size_tensor), (input_tensor, window_size_tensor), MaxPool)
