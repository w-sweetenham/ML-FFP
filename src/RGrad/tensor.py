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

    def backward(self):
        if self.shape != ():
            raise ValueError('Tensor is not a scalar')
        meta_graph = {self.tensor_index: {'children': []}}
        child_list = []
        parent_list = [self]
        contin = True
        while(contin):
            child_list = parent_list
            parent_list = []
            for child_tensor in child_list:
                for parent in child_tensor.parents:
                    if parent.tensor_index in meta_graph:
                        meta_graph[parent.tensor_index]['children'].append(child_tensor.tensor_index)
                    else:
                        meta_graph[parent.tensor_index] = {'children': [child_tensor.tensor_index]}
                    if parent not in parent_list:
                        parent_list.append(parent)