from CamillaGrad.tensor import Tensor

class Optimizer():

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad() # sets the gradient to be 0 so that it doesnt accumulate over runs.

    def step(self):
        raise NotImplementedError()
    
    