import numpy as np

from src.ArtificialDatasets.gaussian_dataset import GaussianDataset

def test_single_class():
    dataset = GaussianDataset([[([1, 1], [[1, 0], [0, 1]])]], 1000)
    assert dataset.num_datapoints == 1000
    assert dataset.relative_probs == [1.0]
    assert sum(dataset.datapoints[:, 0] == np.zeros(1000)) == 1000

def test_2_class():
    dataset = GaussianDataset([[([1, 1], [[1, 0], [0, 1]])], [([-1, 1], [[1, 0], [0, 1]])]], 1000, [0.5, 0.5])
    assert dataset.num_datapoints == 1000
    assert dataset.relative_probs == [0.5, 0.5]
    num_zero_index = sum(dataset.datapoints[:, 0] == np.zeros(1000))
    num_one_index = sum(dataset.datapoints[:, 0] == np.ones(1000))
    assert abs(num_zero_index - num_one_index) <= 200
    assert num_zero_index + num_one_index == 1000

def test_multi_cluster_classes():
    class_0_clusters = [([1, 1], [[1, 0], [0, 1]]), ([-1, -1], [[1, 0], [0, 1]])]
    class_1_clusters = [([-1, 1], [[1, 0], [0, 1]]), ([1, -1], [[1, 0], [0, 1]])]
    dataset = GaussianDataset([class_0_clusters, class_1_clusters], 10000)
    assert dataset.num_datapoints == 10000
