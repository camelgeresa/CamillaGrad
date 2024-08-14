from .optimizer import Optimizer
from ..tensor import Tensor
import numpy as np

class Adam(Optimizer):
    def __init__(self, parameters, learning_rate, beta1=0.9, beta2=0.999):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1 # how much the previous gradients contribute to updating the parameter
        self.beta2 = beta2 # the contribution from the sum of previous gradients squared
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.s = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t +=1
        for i,p in enumerate(self.parameters):
            mom = self.m[i]
            vel = self.s[i]
            
            mom = self.beta1 * mom + (1 - self.beta1) * p.grad
            mom_t = mom/(1-self.beta1**self.t) 
            
            vel = self.beta2*vel + (1-self.beta2)*(p.grad**2)
            vel_t = vel/(1-self.beta2**self.t)

            p.data -= self.learning_rate * mom_t / (np.sqrt(vel_t) + 1e-8)


            self.m[i] = mom
            self.s[i] = vel

