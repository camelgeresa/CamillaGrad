'''We need to test the following operations:
- Addition
- Subtraction 
- Multiplication (broadcasted & non-broadcasted)
- Division
- Matrix multiplication
- exp
- powers
- relu 

Additionally I want to make sure that when a tensor appears several times in the computational graph, all the gradient contributions sum up.'''

import torch
from src.CamillaGrad.tensor import Tensor
import numpy as np

x = Tensor(np.array([[1,2,3],[4,5,6]]))
y = Tensor(np.array([[1,2,3]]))
z = Tensor(np.array([1,1,1]))
x2 = Tensor(np.array([[2,3],[5,6],[7,8]]))

x_t = torch.tensor(x.data, dtype=float, requires_grad=True)
y_t = torch.tensor(y.data, dtype=float, requires_grad=True)
z_t = torch.tensor(z.data, dtype=float, requires_grad=True)
x2_t = torch.tensor(x2.data, dtype=float, requires_grad=True)

def perform_ops(a,b,c):
    d = a+b # Broadcast
    d.retain_grad()
    e = a-b
    e.retain_grad()
    f = d*e
    f.retain_grad()
    g = f/c
    g.retain_grad()
    h = g.exp()
    h.retain_grad()
    i = h.relu()
    i.retain_grad()
    k = i.mean()
    
    k.backward()

    return [a.grad, b.grad, c.grad, d.grad, e.grad, f.grad, g.grad, h.grad, i.grad, k.grad]

def perform_ops2(a,b):
    c = a@b
    c.retain_grad()
    d = c**3
    d.retain_grad()
    e = d.sum()

    e.backward()
    return [a.grad, b.grad, c.grad, d.grad, e.grad]

grads1 = perform_ops(x,y,z)
grads2 = perform_ops2(x,x2)
grads_true1 = perform_ops(x_t, y_t, z_t)
grads_true2 = perform_ops2(x_t, x2_t)


def test_compare():
    for i,j in zip(grads1, grads_true1):
        try:
           assert np.allclose(i, j.detach().numpy())
        except:
            assert j == None
    for i,j in zip(grads2, grads_true2):
        try:
           assert np.allclose(i, j.detach().numpy())
        except:
            assert j == None






