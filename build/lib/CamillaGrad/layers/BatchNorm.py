from ..tensor import Tensor
import numpy as np
from .. import Module

class BatchNorm(Module):
    def __init__(self, num_feats):
        super().__init__()
        self.eps = 1e-10
        self.gamma = 0.01*Tensor(np.random.randn(num_feats))
        self.beta = 0.01*Tensor(np.random.randn(num_feats))

    def forward(self,x):
        BS, C, H,W = x.shape
        self._cache['x'] = x
        mu = x.mean(axis = (0,2,3), keepdims=True) # find mean across batch
        var = Tensor(np.std(x.data, axis=(0,2,3), keepdims=True))
        x_hat = (x-mu)/(var + self.eps)**0.5

        self._cache['mu'] = mu.data
        self._cache['var'] = var.data
        self._cache['x_hat'] = x_hat.data

        # Gamma & beta should be broadcasted!
        output =  self.gamma[None,:,None,None] * x_hat + self.beta[None,:,None,None]
        output = Tensor(output.data, (x,), backward_ = self.backward)
        self._cache['output'] = output
        return output

    def backward(self):
        dout = self._cache['output'].grad
        x_tenser = self._cache['x']
        x = x_tenser.data
        BS = x.shape[0]
        x_hat = self._cache['x_hat']
        mu = self._cache['mu']
        var = self._cache['var']

        dBeta = np.sum(dout, axis=(0,2,3)) # Broadcasted across multiple dims
        dGamma = np.sum(dout * x_hat, axis = (0,2,3)) # gets rid of row dim.
        
        dx_hat = dout*self.gamma.data[None,:,None,None]

        # dvar has shape [1 x F], a row vector. We broadcast it to add rows.
        # we also need to broadcast gamma.
        # but then we also need to sum over batch dim.
        dVar = np.sum(dx_hat * (x - mu) * -0.5 * (var + self.eps)**-1.5, axis=(0, 2, 3), keepdims=True)

        dMu_from_xhat = dx_hat*-1*(var + self.eps)**-0.5
        dMU_from_var = dVar * np.sum(-2*(x - mu),(0,2,3), keepdims=True)/ (BS+1)
        dMu = np.sum(dMu_from_xhat, axis=(0,2,3),keepdims=True) + dMU_from_var # 2nd part -> same batch get same mu, so we sum over it.
        
        # Downstream gradient
        dx = dx_hat*(var-self.eps)**-0.5 +  dVar*2*(x - mu)*1/(BS+1) + dMu/BS
        # nicer -> we don't have to worry about broadcasting since its wrt to x_hat (which is not broadcatsed along batch dim as mu/var is)

        self.gamma.grad = dGamma
        self.beta.grad = dBeta
        x_tenser.grad = dx
