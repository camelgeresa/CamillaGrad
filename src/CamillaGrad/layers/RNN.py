from ..tensor import Tensor
from ..module import Module
import numpy as np

class RNN(Module):
    def __init__(self,num_features, hidden_dim, output_dim):
        super().__init__()
        #self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.w_xh = Tensor(np.random.randn(num_features, self.hidden_dim)) # F -> H
        self.w_hh = Tensor(np.random.randn(self.hidden_dim, hidden_dim))
        self.w_hy = Tensor(np.random.randn(self.hidden_dim, output_dim))
        self.bh = Tensor(np.zeros(self.hidden_dim,))
        self.by = Tensor(np.zeros(output_dim,))

    def forward(self, x):
        # x is of shape BS x T x F
        BS, T, F = x.shape
        outputs = []
        hidden_states = []
        h = Tensor(np.zeros(self.hidden_dim))

        for t in range(T):
            h =  (x[:,t] @ self.w_xh + h @ self.w_hh + self.bh).tanh()
            hidden_states.append(h)

            output = h @ self.w_hy + self.bh
            outputs.append(output)

        outputs = Tensor(np.stack([o.data for o in outputs]))
        
        self._cache['outputs'] = outputs
        self._cache['hidden_states'] = hidden_states
        self._cache['x'] = x

        return outputs, hidden_states
    

    def backward(self, dout):
        # dout -> list of outs:
        dw_xh, dw_hh, dw_yh = Tensor(np.zeros_like(self.w_xh)), Tensor(np.zeros_like(self.w_hh)), Tensor(np.zeros_like(self.w_yh))
        dbh, dby = Tensor(np.zeros_like(self.bh)), Tensor(np.zeros_like(self.by))

        hidden_states = self._cache['hidden_states']
        dh_next = Tensor(np.zeros_like(hidden_states[0]))

        for i in range(len(dout),0,-1):
            dout_t = dout[i]
            dby += dout_t.sum(axis=0)

            dw_yh += hidden_states[i].transpose() @ dout_t

            dh = dout_t @ self.w_hy + dh_next 
            dh_raw = dh*(1-hidden_states[i]**2)

            dbh += dh_raw.sum(axis=0)

            dw_hh += hidden_states[i-1].transpose() @ dh_raw 
            dw_xh += self._cache['x'][i].transpose() @ dh_raw

            dh_next = dh_raw @ self.w_hh.transpose()

        self._grads['dw_xh'] = dw_xh
        self._grads['dw_hh'] = dw_hh
        self._grads['dw_yh'] = dw_yh
        self._grads['db_h'] = dbh
        self._grads['db_y'] = dby

        return None








