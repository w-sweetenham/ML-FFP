"""module providing classes for different artificial datasets"""
import numpy as np


class GaussianDataset:
    """
    Represents a dataset of values chosen each corresponding to a different
    class. There is a certain probability that each datapoint is selected from
    each class. A class corresponds to a set of Gaussian pdfs and a datapoint
    sampled from that class has equal probability of being sampled from each
    distribution in that class.
    """

    def __init__(self, class_specs, num_samples, relative_probs=None):
        """
        Summary:
            Instantiates the dataset.

        Args:
            class_specs (list): A list of lists. Each second-level list
                corresponds to a class and should contain a sequence of tuples
                each corresponding to an individual Gaussian distribution. The
                first element of the tuple should be the mean of the
                distribution and the second element should be the covariance
                matrix.
            num_samples (int): The total number of samples in the dataset,
                spread across each class
            relative_probs (list, optional): A list with length the same as
                the number of classes. The values should form a probability
                distribution as they are used to determine the relative
                probability of samples being chosen from each class
        """
        if relative_probs is None:
            self.relative_probs = [1 / len(class_specs)] * len(class_specs)
        else:
            self.relative_probs = relative_probs
        self.num_datapoints = num_samples
        self.class_specs = class_specs
        class_indexes = np.random.choice(
            np.arange(len(class_specs)), num_samples, self.relative_probs
        )
        self.datapoints = {"value": [], "class": []}
        for class_index, class_spec in enumerate(self.class_specs):
            num_samples_for_class = sum(class_indexes == class_index)
            gaussian_indexes = np.random.choice(
                np.arange(len(class_spec)), num_samples_for_class
            )
            for gaussian_index, gaussian_spec in enumerate(class_spec):
                num_samples_for_gaussian = sum(
                    gaussian_indexes == gaussian_index
                )
                gaussian_datapoints = np.random.multivariate_normal(
                    gaussian_spec[0],
                    gaussian_spec[1],
                    num_samples_for_gaussian,
                )
                indexes = (
                    np.ones((num_samples_for_gaussian,), dtype=np.int32)
                    * class_index
                )
                self.datapoints["value"].append(gaussian_datapoints)
                self.datapoints["class"].append(indexes)
        self.datapoints["value"] = np.concatenate(self.datapoints["value"])
        self.datapoints["class"] = np.concatenate(self.datapoints["class"])
        shuffled_indexes = np.random.permutation(len(self.datapoints["value"]))
        self.datapoints["value"] = self.datapoints["value"][shuffled_indexes]
        self.datapoints["class"] = self.datapoints["class"][shuffled_indexes]
