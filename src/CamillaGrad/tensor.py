import numpy as np

class Tensor:
  def __init__(self, data, children = (), backward_ = lambda:None):
    self.data = data # should be an np array
    self.children = set(children)
    self.grad = 0 # should be an np array
    self.backward_ = backward_
    self.requires_grad = True
    self.shape = self.data.shape

  def __repr__(self):
    return f'Tensor(data = {self.data}, grad = {self.grad})'
  
  def ensure_tensor(self, t):
    if isinstance(t,Tensor):
      return t
    elif isinstance(t, np.ndarray):
      return Tensor(t)
    else:
      return Tensor(np.array([t]))

  def __matmul__(self, other):
    other = self.ensure_tensor(other)
    out = Tensor(self.data @ other.data, (self, other))

    def backward_():
      self.grad += out.grad@ other.data.T
      other.grad += self.data.T @ out.grad

    out.backward_ = backward_
    return out

  def __add__(self, other):
    other = self.ensure_tensor(other)
    out = Tensor(self.data + other.data, (self, other))

    def backward_():
       temp_self_grad, temp_other_grad = out.grad.copy(), out.grad.copy()
       # if we had to broadcast for summation, we'll need to add the contributions.
       dims_added_self = out.grad.ndim - self.data.ndim
       for _ in range(dims_added_self):
        temp_self_grad = temp_self_grad.sum(axis = 0) # when broadcasting 1 is added to the start of shape.

       # In the case we do already have the same ndim but its a 1
       for i,dim in enumerate(self.data.shape):
          if dim == 1:
            # we need to sum along this dim because it was broadcasted along this dim, i.e. this element contributed multiple times along this dim
            temp_self_grad = temp_self_grad.sum(axis = i, keepdims=True)


       dims_added_other = out.grad.ndim - other.data.ndim
       for _ in range(dims_added_other):
          temp_other_grad = temp_other_grad.sum(axis = 0) # when broadcasting 1 is added to the start of shape.

        # In the case we do already have the same ndim but its a 1
       for i,dim in enumerate(other.shape):
          if dim==1:
            # we need to sum along this dim because it was broadcasted along this dim, i.e. this element contributed multiple times along this dim
            temp_other_grad = temp_other_grad.sum(axis = i, keepdims=True)
       
       self.grad += temp_self_grad
       other.grad += temp_other_grad


    out.backward_ = backward_
    return out
  
  def __sub__(self,other):
    other = self.ensure_tensor(other)
    return self + (-other)
  
  def __rsub__(self, other):
    other = self.ensure_tensor(other)
    return other - self
  
  def __truediv__(self,other):
    other = self.ensure_tensor(other)
    return self*(other**-1)
  
  def __rtruediv__(self,other):
    other = self.ensure_tensor(other)
    return other.__truediv__(self)
  
  def __mul__(self, other):
    other = self.ensure_tensor(other)
    out = Tensor(self.data * other.data, (self, other))

    def backward_():
      # c = a*b where b is broadcasted
      # When diff wrt non-broadcasted element, that will be out.grad * broadcasted b -> but thats taken care for us.

      temp_self_grad = out.grad * other.data
      temp_other_grad = out.grad * self.data

      dims_added_self = out.grad.ndim - self.data.ndim
      for _ in range(dims_added_self):
        temp_self_grad = temp_self_grad.sum(axis = 0)

      for i,dim in enumerate(self.data.shape):
        if dim == 1:
          # we need to sum along this dim because it was broadcasted along this dim, i.e. this element contributed multiple times along this dim
          temp_self_grad = temp_self_grad.sum(axis = i, keepdims=True)


      dims_added_other = out.grad.ndim - other.data.ndim
      for _ in range(dims_added_other):
        temp_other_grad = temp_other_grad.sum(axis = 0) # when broadcasting 1 is added to the start of shape.

      # In the case we do already have the same ndim but its a 1
      for i,dim in enumerate(other.shape):
        if dim == 1:
          # we need to sum along this dim because it was broadcasted along this dim, i.e. this element contributed multiple times along this dim
          temp_other_grad = temp_other_grad.sum(axis = i, keepdims=True)

      self.grad += temp_self_grad
      other.grad += temp_other_grad

    out.backward_ = backward_
    return out

  def __rmul__(self, other):
    other = self.ensure_tensor(other)
    return self*other

  def sum(self, axis=None, keepdims=False):
    out = Tensor(self.data.sum(axis = axis, keepdims=keepdims), (self,))

    def backward_():
      # we need to broadcast the gradient, because each element will contribute a 1.
      self.grad += np.ones_like(self.data)* out.grad # out.grad will need to be broadcast.

    out.backward_ = backward_
    return out
  
  def __pow__(self, power):
    out = Tensor(self.data.astype(float)**power, (self,))

    def backward_():
      self.grad += out.grad*power*self.data.astype(float)**(power-1)

    out.backward_ = backward_
    return out
  
  def __neg__(self):
    return self*-1

  def __radd__(self, other):
    return self+other
  
  def __iadd__(self, other):
    return self+other
  
  def exp(self):
    out = Tensor(np.exp(self.data), (self,))

    def backward_():
      self.grad += out.data*out.grad

    out.backward_ = backward_
    return out
  
  def log(self):
    out = Tensor(np.log(self.data))

    def backward_():
      self.grad += out.grad * 1/out

    out.backward_ = backward_
    return out

  def __getitem__(self, idx):
    out = Tensor(self.data[idx], (self,))

    def backward_():
      temp = np.zeros_like(self.data)
      temp[idx] = out.grad[idx] #1
      self.grad += temp

    out.backward_ = backward_
    return out
  
  def __setitem__(self,idx,val):
    if isinstance(idx, Tensor):
      idx = idx.data
    self.data[idx] = val.data
    out = Tensor(self.data, (self,))
    
    def backward_():
      raise NotImplementedError()

    return out
  
  def mean(self, axis=None, keepdims = False):
    out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,))

    def backward_():
      # scaled and broadcasted.
      s = slice(axis)
      scale_factor = np.prod(self.data.shape[s])
      self.grad += np.ones_like(self.data)*out.grad*1/scale_factor

    out.backward_ = backward_
    return out
  
  def softmax(self):
    exp_x = np.exp(self.data - np.max(self.data, axis=1, keepdims=True)) # axis=1 -> sum along the cols (classes)
    out = Tensor(exp_x / np.sum(exp_x, axis=1, keepdims=True), (self,))

    def backward_():
      self.grad += out.data*(out.grad - out.grad*out.data.sum(axis=1, keepdims=True))

    out.backward_ = backward_
    return out

  def relu(self):
    out = Tensor(np.maximum(0,self.data), (self,))

    def backward_():
      self.grad += np.where(out.data > 0, out.grad, 0)

    out.backward_ = backward_
    return out
  
  def sigmoid(self):
    return Tensor(1/(1 + np.exp(-self.data)))

    def backward_():
      raise NotImplementedError()

  def concat_(self, other, axis):
    # other is a list of elements of type Tensor
    # concat only works for axis that already exists!
    tensors = [self]+ other if isinstance(other,list) else [self,other]
    concat_items = [x.data for x in tensors]
    out = Tensor(np.concatenate(concat_items, axis = axis), (self, other))

    def backward_():
      # Adapted from https://github.com/laksjdjf/dezero-diffusion/blob/a223c7e2bb06e149ff0a8b0714fcc88fb38b08b7/modules/unet.py#L10-L40
      start_idx = 0

      for t, c in zip(tensors, concat_items):
         end_idx = start_idx + c.shape[axis] # the axis component is the only one that changes
         indices = [slice(None)] * out.grad.ndim
         indices[axis] = slice(start_idx, end_idx, None) # ensures that we slice along the correct dim
         t.grad += out.grad[tuple(indices)]
         start_idx = end_idx

    out.backward_ = backward_
    return out
  
  def transpose(self, axis= None):
    # axis -> tuple or list of ints. A permutation of the dims.
    out = Tensor(np.transpose(self.data, axes = axis), (self,))
    
    def backward_():
      # Transpose opposite
      if axis is not None:
        self.grad += np.transpose(out.grad, axes = np.argsort(axis))
      else:
        self.grad += np.transpose(out.grad)

    out.backward_ = backward_
    return out
  
  def pad(self, pad_width):
    # pad_width -> tuples((amount to pad on top, amount to pad on the bottom),(amount to pad on left, amount to pad on the right))
    out = Tensor(np.pad(self.data, pad_width), (self,))

    def backward_():
      if not isinstance(pad_width, tuple):
        pad_width = (pad_width,)*self.data.ndim
      self.grad = out.grad
      for i in range(out.grad.ndim):
          length = out.grad.shape[i]
          slices = [slice(None, None,None)]*out.grad.ndim
          slices[i] = slice(2,length-2,None)
          self.grad =self.grad[tuple(slices)]

    out.backward_ = backward_
    return out
    
  def reshape(self, newshape):
    out = Tensor(np.reshape(self.data, newshape), (self,))

    def backward_():
      self.grad += np.reshape(out.grad, self.shape)

    out.backward_ = backward_
    return out
  
  def tanh(self):
    out = Tensor(np.tanh(self.data), (self,))

    def backward_():
      self.grad += out.grad*(1-out.data**2)
    
    out.backward_ = backward_
    return out
  

  def backward(self, grad = None):

    if not self.requires_grad: raise RuntimeError('Backward has been called for tensor which has requires grad set to false')

    if grad is None:
      if self.shape == (): # if self is a scalar, for example the loss.
        self.grad= np.array([1])

      else:
        raise RuntimeError('grad must be specified for non scalar!') 

    else: 
      self.grad = grad.data

    seen = []
    nodes = []
    def topo(node):
      # We topologically sort 
      if node not in seen:
        seen.append(node)
        for child in node.children:
            topo(child)
        # The leaves will be added here first
        nodes.append(node)

    topo(self)
    
    for node in nodes[::-1]:
      node.backward_() 

  def retain_grad(self):
      pass
    
  def zero_grad(self):
    self.grad = 0