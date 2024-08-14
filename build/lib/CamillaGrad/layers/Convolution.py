import numpy as np
from ..module import Module
from ..tensor import Tensor
from .utils import Im2Col, Col2Im


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel_weights = 0.01*Tensor(np.random.randn(out_channels,in_channels, *kernel_size))
        self.kernel_bias = Tensor(np.zeros(out_channels)) # we add a bias for each channel

    def forward(self, x):
        self._cache['x'] = x
        out_channels, in_channels, kernel_h, kernel_w = self.kernel_weights.shape
        # W_out = (W_in - K + 2P)/S + 1 
        N, C, H, W = x.shape
        H_out = int((H - kernel_h + 2*self.padding)/ self.stride + 1)
        W_out = int((W - kernel_w + 2*self.padding)/ self.stride + 1)

        col_weights = self.kernel_weights.reshape((out_channels , -1)).transpose() #[patch,K]
        col = Im2Col(x, kernel_h, kernel_w, self.stride, self.padding) # [convs x patches]

        output = col @ col_weights + self.kernel_bias # convs x num kernels
        output =  output.reshape((N, H_out, W_out, -1)).transpose((0, 3, 1, 2))

        self._cache['x_col'] = col.data
        self._cache['col_weights'] = col_weights.data

        output = Tensor(output.data, (x,), backward_ = self.backward)
        self._cache['output'] = output

        return output

    def backward(self):
        # output shape -> N,K, Hout, Wout
        dout = self._cache['output'].grad # this should have been updated from the next layer.
        x = self._cache['x']
        
        dout = dout.transpose(0,2,3,1).reshape((-1, self.out_channels)) # [N, Hout, Wout, K] -> [N*Hout*Wout, K] -> [convs, K]
        db = dout.sum(axis = 0) # we sum along the N*Hout*Wout axis -> (K,)
        #print(f'{dout.shape=}, {self._cache['x_col'].shape=}')
        dw_col = dout.transpose() @ self._cache['x_col'] # [K x convs] x [convs x patch] -> [K x patch], the 2nd one (dout) is the filter.
        dw = dw_col.transpose(1,0).reshape((self.out_channels, self.in_channels, *self.kernel_size))
        dx_col = dout @ self._cache['col_weights'].transpose() # [convs, K] x [K, patch]
        dx = Col2Im(dx_col, self._cache['x'].shape, *self.kernel_size, self.stride, self.padding)

        self.kernel_bias.grad = db
        self.kernel_weights.grad = dw
        x.grad = dx # this should be used downstream

        
    


class MaxPool(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        N, C, H, W = x.shape
        assert self.stride == self.kernel_size[0] == self.kernel_size[1] # this method only works for non-overlapping boxes. 

        out_h = int(1 + (H - self.kernel_size[0]) / self.stride)
        out_w = int(1 + (W - self.kernel_size[1]) / self.stride)

        boxes = x.reshape((N, C, H//self.kernel_size[0], self.kernel_size[0], W//self.kernel_size[1], self.kernel_size[1]))

        output = boxes.data.max(axis=3).max(axis=4) # be careful np array not tensor

        self._cache['x'] =x
        self._cache['boxes']= boxes.data

        output = Tensor(output, (x,), backward_ = self.backward)
        self._cache['output'] = output

        return output
    
    def backward(self):
        # https://github.com/madalinabuzau/cs231n-convolutional-neural-networks-solutions/blob/master/2017%20Spring%20Assignments/assignment3/cs231n/fast_layers.py
        # dout shape -> N x C x H_new x W_new
        output = self._cache['output']
        dout = output.grad
        boxes = self._cache['boxes']
        x_tensor = self._cache['x']
        x = x_tensor.data

        # find the indices where the max has occured. Only replace those with the gradient.
        x_reshaped = boxes
        out = output.data

        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx = dx_reshaped.reshape(x.shape)
        
        x_tensor.grad = dx

        




        







        






        


        


        

