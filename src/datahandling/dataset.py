import numpy as np


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
            image_array = np.array(image_dims)
            for row_num in range(image_dims[0]):
                image_array[row_num] = pixel_vals[row_num*image_dims[1]:row_num*image_dims[1] + image_dims[1]]
            self.datapoints.append((image_array, label))
