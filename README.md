## Algorithms From First Principles
This repo contains code implementing a deep learning framework from first principles which can be used to define and train deep learning models with minimal code.

The only external library used in the core code is Numpy which performs most of the large computations.

## Repo Struture
There are 4 top-level directories in the repo:
* **datasets** which contains test data for the unit tests and to which the mnist data should be saved in order to run the example notebook (see the next section)
* **notebooks** containing the example notebook demonstrating how to define and train a fully connected network on the mnist dataset
* **src** which contains all of the source code for the framework
* **test** which contains a set of unit tests written using the Pytest framework

## Executing the Example Notebook
### Creating the environment
Firstly, the virtual environment within which the notebook is to be run must be created. This can be done by running:
```
python3 -m venv ./env
source env/bin/activate
pip install -r requirements.txt
```
at the top level of the repo.

### Downloading the data
The notebook looks in the datasets top level directory for 2 files: mnist_train.csv and mnist_test.csv. These can be downloaded from: [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)

### Running the notebook
To start up the notebook, execute:
```
cd datasets
jupyter notebook
```
from the top level of the repo. Note that the notebook adds the parent of the current working directory to the path in order to find the src directory and so it's important that the notebook is executed from within the notebooks directory.

Once the notebook has started up, you can just run all the cells and it should train the model and present its accuracy and display how the loss varies over the batches. Note that currently the model is set to only train for a single epoch as it around 15 mins to complete one epoch on the train set.

### Code Structure
The 2 core concepts on which the rest of the code is based are Tensors and Operations. A Tensor here corresponds to the mathematical idea of a tensor and implemented as a class which contains the actual data in a numpy array and has pointers to a class representing the operation from which it was created as well as the tensors fed into that operation to create it (these pointers are set to None if it was not created from other tensors). Each operation has a class with static methods for its forward and backward passes. The forward pass method takes in a list of tensors and returns a numpy array of the resulting tensor. The backward pass method is a generator which takes in the same input tensors as the forward method as well as an index specifying which of those inputs to get the derrivatives with respect to. It then yields, for each position in the output tensor, an index specifying the position and the tensor of partial derrivatives of that output element with respect to the specified input tensor. The operation is also associated with an operation function which takes in the relevant tensors and creates a new tensor (using the forward pass of the operation) with parents as the tensors which were passed in as arguments. Tensor objects have a backward() method which can only be called on scalar tensors. This method determines the graph of tensors which formed the current scalar tensor and for each of them, computes a tensor of partial derrivatives of that tensor with respect to the scalar tensor.

Built on top of the tensor and operation concepts is the idea of a transform and a parameter. A transform corresponds to the idea of a parameterized function and a parameter corresponds to the parameters controlling that function. A parameter is implemented as a class wrapping a tensor and a transform is implemented as a class which contains parameters as well as other transforms. There is a base Transform class which defines a method to recursively return all the parameters both from that transform object as well as from all the transforms objects it contains. To create new transforms, this base transform class can be inherited from. The new class requires __init__ and __call__ methods to be defined. The __call__ method should take in the tensor inputs to the transform and return a new tensor created by calling operation functions on those input tensors and the parameters of the transform. As transforms can be nested, complicated neural networks comprising large numbers of individual transforms which may themselves be formed from other transforms, can be created as transforms.

Stochastic gradient descent and momentum optimizers have also been implemented to help with training. These are each initialized with the parameters they should be optimizing and implement a step method to update the parameters based on the gradients (and the current state of the optimizer in the case of momentum).

A class has been created for representing a greyscale image dataset stored as a csv document (one element per pixel, all pixels of a given image on one row). A dataloader class which can batch up individual examples from the dataset class has also been provided.