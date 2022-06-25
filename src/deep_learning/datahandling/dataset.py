import numpy as np

from src.deep_learning.RGrad.tensor import Tensor


class LabelledDataset:

    def __init__(self):
        self.datapoints = []

    def num_datapoints(self):
        return len(self.datapoints)

    def get_datapoint(self, index):
        return self.datapoints[index]


class CSVImageDataset(LabelledDataset):

    def __init__(self, path, image_dims):
        super().__init__()
        num_pixels = image_dims[0]*image_dims[1]
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            split_line = [val for val in line[:-1].split(',')]
            pixel_vals = [float(val)/255 for val in split_line[1:]]
            label = int(split_line[0])
            if len(pixel_vals) != num_pixels:
                raise ValueError('invalid line length')
            image_array = np.zeros(image_dims)
            for row_num in range(image_dims[0]):
                image_array[row_num] = pixel_vals[row_num*image_dims[1]:row_num*image_dims[1] + image_dims[1]]
            self.datapoints.append((image_array, label))


class DataLoader:

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        num_datapoints = self.dataset.num_datapoints()
        self.index_sequence = np.arange(num_datapoints)
        if self.shuffle:
            np.random.shuffle(self.index_sequence)
        num_full_batches = num_datapoints // self.batch_size
        self.index_sequence_tuples = []
        for batch_num in range(num_full_batches):
            self.index_sequence_tuples.append((batch_num*self.batch_size,(batch_num*self.batch_size) + self.batch_size))
        if num_full_batches*self.batch_size < num_datapoints:
            self.index_sequence_tuples.append((num_full_batches*self.batch_size, num_datapoints))

        self.index_tuples_index = 0
        return self

    def __next__(self):
        if self.index_tuples_index < len(self.index_sequence_tuples):
            datapoint_indices = self.index_sequence[self.index_sequence_tuples[self.index_tuples_index][0]:self.index_sequence_tuples[self.index_tuples_index][1]]
            self.index_tuples_index += 1
            image_list = []
            label_list = []
            for index in datapoint_indices:
                image, label = self.dataset.get_datapoint(index)
                image_list.append(image)
                label_list.append(label)
            return Tensor(np.stack(image_list)), Tensor(np.array(label_list))
        else:
            raise StopIteration
