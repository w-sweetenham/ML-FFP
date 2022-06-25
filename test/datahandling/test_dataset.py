import os
import numpy as np

from src.deep_learning.datahandling.dataset import CSVImageDataset, DataLoader


def test_csv_image_dataset():
    dataset = CSVImageDataset(os.path.join('datasets', 'test_image_dataset.csv'), (2, 3))
    im1_correct = np.array([[0, 5/255, 1], [253/255, 0, 7/255]])
    label1_correct = 0
    im2_correct = np.array([[254/255, 1, 0], [0, 1/255, 0]])
    label2_correct = 7

    im1, label1 = dataset.get_datapoint(0)
    im2, label2 = dataset.get_datapoint(1)

    assert np.allclose(im1, im1_correct)
    assert label1 == label1_correct
    assert np.allclose(im2, im2_correct)
    assert label2 == label2_correct


def test_dataloader():
    dataset = CSVImageDataset(os.path.join('datasets', 'test_image_dataset.csv'), (2, 3))
    dataloader = DataLoader(dataset, 2)
    image_batches = []
    label_batches = []
    for image_batch, label_batch in dataloader:
        image_batches.append(image_batch)
        label_batches.append(label_batch)
    
    image_batch1_correct = np.array([[[0, 5/255, 1], [253/255, 0, 7/255]], [[254/255, 1, 0], [0, 1/255, 0]]])
    label_batch1_correct = np.array([0, 7])
    image_batch2_correct = np.array([[[8/255, 8/255, 254/255], [252/255, 4/255, 253/255]], [[251/255, 254/255, 1], [1, 1/255, 2/255]]])
    label_batch2_correct = np.array([1, 9])
    image_batch3_correct = np.array([[[250/255, 187/255, 100/255], [0, 0, 0]]])
    label_batch3_correct = np.array([8])

    assert np.allclose(image_batches[0].elems, image_batch1_correct)
    assert np.all(label_batches[0].elems == label_batch1_correct)
    assert np.allclose(image_batches[1].elems, image_batch2_correct)
    assert np.all(label_batches[1].elems == label_batch2_correct)
    assert np.allclose(image_batches[2].elems, image_batch3_correct)
    assert np.all(label_batches[2].elems == label_batch3_correct)
    