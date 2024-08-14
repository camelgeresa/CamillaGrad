from ..module import Module
from ..tensor import Tensor
import numpy as np

class Linear(Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        a = 1. / np.sqrt(self.dim2)
        self.weight = Tensor(np.random.uniform(-a, a, size=(self.dim1, self.dim2)))
        self.bias = Tensor(np.zeros((self.dim2)))

    def forward(self,x):
        self._cache['x'] = x
        return x @self.weight + self.bias
    
    def backward(self, dloss):
        self._grads['w'] = self._cache['x'].transpose() @ dloss
        self._grads['b'] = dloss.sum(axis = 0)
        dx = dloss @ self.weight.transpose()

        return dx


        