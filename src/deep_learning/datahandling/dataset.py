"""Module providing dataset classes and data loaders"""
import numpy as np

from src.deep_learning.RGrad.tensor import Tensor


class LabelledDataset:
    """
    superclass for dataset classes
    """

    def __init__(self):
        self.datapoints = []

    def num_datapoints(self):
        """
        returns number of datapoints in dataset

        Returns:
            int: number of datapoints
        """
        return len(self.datapoints)

    def get_datapoint(self, index):
        """
        returns a given a given datapoint at the specified index

        Args:
            index (int): index of datapoint

        Returns:
            : the datapoint
        """
        return self.datapoints[index]


class CSVImageDataset(LabelledDataset):
    """
    class representing a dataset of images stored as a single csv file
    """
    def __init__(self, path, image_dims):
        """
        Args:
            path (str): string of path to csv file
            image_dims (tuple): tuple of dimensions of image

        Raises:
            ValueError: if number of pixel cols in csv not same as product of
                given image dims
        """
        super().__init__()
        num_pixels = image_dims[0] * image_dims[1]
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines[1:]:
            split_line = line[:-1].split(",")
            pixel_vals = [float(val) / 255 for val in split_line[1:]]
            label = int(split_line[0])
            if len(pixel_vals) != num_pixels:
                raise ValueError("invalid line length")
            image_array = np.zeros(image_dims)
            for row_num in range(image_dims[0]):
                image_array[row_num] = pixel_vals[
                    row_num * image_dims[1] : row_num * image_dims[1]
                    + image_dims[1]
                ]
            self.datapoints.append((image_array, label))


class DataLoader:
    """
    class to represent a data loader which batches together individual items in
        and iterates through them
    """
    def __init__(self, dataset, batch_size, shuffle=False):
        """
        Args:
            dataset (LabelledDataset): dataset to batch and iterate through
            batch_size (int): size of batches
            shuffle (bool, optional): whether to shuffle order of iteration
                through dataset
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.index_sequence = None
        self.index_tuples_index = None
        self.index_sequence_tuples = None

    def __iter__(self):
        num_datapoints = self.dataset.num_datapoints()
        self.index_sequence = np.arange(num_datapoints)
        if self.shuffle:
            np.random.shuffle(self.index_sequence)
        num_full_batches = num_datapoints // self.batch_size
        self.index_sequence_tuples = []
        for batch_num in range(num_full_batches):
            self.index_sequence_tuples.append(
                (
                    batch_num * self.batch_size,
                    (batch_num * self.batch_size) + self.batch_size,
                )
            )
        if num_full_batches * self.batch_size < num_datapoints:
            self.index_sequence_tuples.append(
                (num_full_batches * self.batch_size, num_datapoints)
            )

        self.index_tuples_index = 0
        return self

    def __next__(self):
        if self.index_tuples_index < len(self.index_sequence_tuples):
            datapoint_indices = self.index_sequence[
                self.index_sequence_tuples[self.index_tuples_index][0]
                : self.index_sequence_tuples[self.index_tuples_index][1]
            ]
            self.index_tuples_index += 1
            image_list = []
            label_list = []
            for index in datapoint_indices:
                image, label = self.dataset.get_datapoint(index)
                image_list.append(image)
                label_list.append(label)
            return Tensor(np.stack(image_list)), Tensor(np.array(label_list))
        raise StopIteration

    def __len__(self):
        return len(self.index_sequence_tuples)
