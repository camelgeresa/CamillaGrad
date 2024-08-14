from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self):
        for p in self.parameters:
            #p = p - self.learning_rate*p.grad
            p.data = p.data - self.learning_rate * p.grad
            