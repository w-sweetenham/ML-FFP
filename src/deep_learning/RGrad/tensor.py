import numpy as np

tensor_index = 0

class Tensor:
    
    def __init__(self, array, parents=None, function=None):
        global tensor_index
        self.elems = array
        self.parents = parents
        self.function = function
        self.shape = array.shape
        self.tensor_index = tensor_index
        self.grad_array = None
        tensor_index += 1

    def copy(self):
        """
        Summary:
            creates a new tensor with elements having the same value as this tensor
        Returns:
            Tensor: copy tensor
        """
        return Tensor(np.copy(self.elems))

    def get_meta_graph(self):
        """
        Summary:
            Computes a dictionary of {tensor_index: <number of children>, ...}

        Raises:
            RuntimeError: If the number of iterations building up the graph exceeds 10,000

        Returns:
            dict: python dictionary specifying the number of children each tensor in the graph has
        """
        meta_graph = {} # a dictionary storing the number of child nodes for each node in the graph
        meta_graph_nodes = {} # a dictionary storing every node in the graph in the form {tensor_index: tensor}

        child_nodes = {self.tensor_index: self}
        parent_nodes = {}
        count = 0
        while count < 10000:
            for tensor_index, child_node in child_nodes.items():
                meta_graph[tensor_index] = 0
                meta_graph_nodes[tensor_index] = child_node
                if child_node.parents is not None:
                    for parent_node in child_node.parents:
                        parent_nodes[parent_node.tensor_index] = parent_node
            if len(parent_nodes) == 0:
                break
            child_nodes = parent_nodes
            parent_nodes = {}
            count += 1
        if count == 10000:
            raise RuntimeError('all 10000 iterations used in building meta-graph')

        for tensor_index, tensor in meta_graph_nodes.items():
            if tensor.parents is not None:
                for parent_tensor in tensor.parents:
                    meta_graph[parent_tensor.tensor_index] += 1
        
        return meta_graph
    
    def add_grad_contribution(self, index):
        """
        Summary:
            assuming this tensor has a grad array of the partial derrivatives
            of itself with respect to some scalar tensor i.e. self.grad_array
            is not None, this computes the partial derrivatives of that scalar
            tensor with respect to the parent at the specified index via this
            tensor. This is done by computing the sum of the multiplications of
            each element of the grad array of this tensor by the tensor of partial
            derrivatives of the parent tensor with respect to the corresponding
            element of this tensor.

        Args:
            index (int): index in self.parents of the parent tensor to add the
            gradient contribution to.
        """
        parent_tensor = self.parents[index]
        derriv_array = np.zeros(parent_tensor.shape)
        backward_tensor = self.function.backward(*self.parents, index)
        if backward_tensor is None:
            return
        for index in np.ndindex(self.shape):
            derriv_array += self.grad_array[index]*backward_tensor[index]
        if parent_tensor.grad_array is None:
            parent_tensor.grad_array = derriv_array
        else:
            parent_tensor.grad_array += derriv_array
    
    def backward(self):
        """
        Summary:
            goes back through the graph of tensors from which this tensor is
            created and computes the gradient tensors of each with respect to
            this one.

        Raises:
            ValueError: if this tensor is not a scalar.
        """
        if self.shape != ():
            raise ValueError('Tensor is not a scalar')
        self.grad_array = np.array(1)
        meta_graph = self.get_meta_graph()
        num_child_grads_added = {}
        gradients_known = {self.tensor_index: self}
        gradients_added = {}
        while len(gradients_known) + len(gradients_added) < len(meta_graph):
            new_gradients_known = {}
            for tensor_index, tensor in gradients_known.items():
                if tensor.parents is not None:
                    for n in range(len(tensor.parents)):
                        tensor.add_grad_contribution(n)
                        parent_tensor_index = tensor.parents[n].tensor_index
                        if parent_tensor_index in num_child_grads_added:
                            num_child_grads_added[parent_tensor_index] += 1
                        else:
                            num_child_grads_added[parent_tensor_index] = 1
                        if num_child_grads_added[parent_tensor_index] == meta_graph[parent_tensor_index]:
                            new_gradients_known[parent_tensor_index] = tensor.parents[n]
                gradients_added[tensor_index] = tensor
            gradients_known = new_gradients_known

    def zero_grads(self):
        """
        Summary:
            sets all the gradient tensors of the tensors from which this tensor
            was created to zeros of the same shape as the tensor. Will do this
            for all tensors including those which currently have no grad
            tensor.
        Raises:
            RuntimeError: If the nodes of the graph aren't computed within
                10,000 iterations.
        """
        meta_graph_nodes = {}

        child_nodes = {self.tensor_index: self}
        parent_nodes = {}
        count = 0
        while count < 10000:
            for tensor_index, child_node in child_nodes.items():
                meta_graph_nodes[tensor_index] = child_node
                if child_node.parents is not None:
                    for parent_node in child_node.parents:
                        parent_nodes[parent_node.tensor_index] = parent_node
            if len(parent_nodes) == 0:
                break
            child_nodes = parent_nodes
            parent_nodes = {}
            count += 1
        if count == 10000:
            raise RuntimeError('all 10000 iterations used in building meta-graph')

        for tensor_index, tensor in meta_graph_nodes.items():
            tensor.grad_array = np.zeros(tensor.shape)
            



                        