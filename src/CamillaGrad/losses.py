from .tensor import Tensor
from .module import Module
import numpy as np

class CrossEntropyLossWithLogits(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
         # logits: shape -> [BS x C]
         # targets: shape of [BS x 1] containing the index of the correct class.
        BS = logits.shape[0]
        probs = logits.softmax()

        log_probs = probs.log()
        out= -log_probs[range(BS),targets.data].mean()
        self._cache['probs'] = probs
        self._cache['targets'] = targets
        self._cache['logits'] = logits
        return Tensor(out.data[0], children=(logits,), backward_ = self.backward)
    
    def backward(self):
        logits = self._cache['logits']
        BS = logits.shape[0]
        targets = self._cache['targets']
        probs = self._cache['probs']
        probs[range(BS),targets.data] -= 1

        logits.grad += probs.data/BS
    

class BinaryCrossEntropyWithLogits(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # logits: shape -> [BS, ] -> logits -> p(y=1|x)
        # targets: shape -> [BS,]
        if len(targets.shape) == 1 and len(logits.shape)==2:
            targets = Tensor(targets.data[:,np.newaxis])

        elif len(targets.shape) == 2 and len(logits.shape)==1:
            logits = Tensor(logits.data[:,np.newaxis])


        probs = logits.sigmoid() # sigmoid

        out = -(targets*(probs.log()) + (1-targets)*((1-probs).log())).mean()
        self._cache['logits'] = logits
        self._cache['targets'] = targets
        return Tensor(out.data[0], children=(logits,), backward_ = self.backward)
    
    def backward(self):
        # dout should = 1
        logits = self._cache['logits']
        targets = self._cache['targets']
        probs = logits.sigmoid()
        logits.grad +=  (probs.data - targets.data)/len(targets.data)








