# All layers must inherit Module and override the forward and backward layers.
from .tensor import Tensor

# E.g class Model(Module):
from collections import OrderedDict
import inspect

class Module:
    def __init__(self):

        self.training = True
        self._cache = OrderedDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    def train(self):
        # when we do model.train(), we calculate the gradients and update the model's parameters.
        self.training = True
        for p in self.parameters():
            # Each p is a tensor
            p.requires_grad = True 

    def eval(self):
        self.training = False
        for p in self.parameters():
            p.requires_grad = False

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                yield value
            if isinstance(value, Module):
                yield from value.parameters()






