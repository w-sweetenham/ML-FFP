from multiprocessing.sharedctypes import Value
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
        tensor_index += 1

    def get_meta_graph():
        meta_graph = {}
        meta_graph_nodes = {}

        child_nodes = {self.tensor_index: self}
        parent_nodes = {}
        count = 0
        while count < 10000:
            for tensor_index, child_node in child_nodes.items():
                meta_graph[tensor_index] = 0
                meta_graph_nodes[tensor_index] = child_node
                for parent_node in child_node.parents:
                    parent_nodes[parent_node.index] = parent_node
            if len(parent_nodes) == 0:
                break
            child_nodes = parent_nodes
            parent_nodes = {}
            count += 1
        if count == 10000:
            raise RuntimeError('all 10000 iterations used in building meta-graph')

        for tensor_index, tensor in meta_graph_nodes.items():
            for parent_tensor in tensor.parents:
                meta_graph[parent_tensor.tensor_index] += 1
        
        return meta_graph
    
    def backward(self):
        if self.shape != ():
            raise ValueError('Tensor is not a scalar')
        meta_graph = self.get_meta_graph()