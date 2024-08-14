import CamillaGrad 
from CamillaGrad.tensor import Tensor
from CamillaGrad.module import Module
from CamillaGrad.layers.Linear import Linear
from CamillaGrad.losses import BinaryCrossEntropyWithLogits
from CamillaGrad.Optimizer.Adam import Adam
import numpy as np

# Creating XOR data
d1 = np.random.rand(50, 2)
# [-1, -1] quarter
d2 = (np.random.rand(50, 2) - 1)
# Labels
l1 = np.ones(len(d1) + len(d2))
l1 = np.column_stack((l1, np.zeros(len(d1) + len(d2))))

# [-1, 1] quarter
d3 = np.random.rand(50, 2)
d3[:,0] -= np.ones(50)
# [1, -1] quarter
d4 = np.random.rand(50, 2)
d4[:,1] -= np.ones(50)
# Labels
l2 = np.zeros(len(d3) + len(d4))
l2 = np.column_stack((l2, np.ones(len(d1) + len(d2))))


# All the data
data = Tensor(np.concatenate((d1, d2, d3, d4)))
labels = np.concatenate((l1, l2)).astype(int)
# labels = np.argmax(labels, axis=1)
labels = Tensor(labels[:,0])
# And all labels :
# [1, 0] one hot encoded: 0 -> blue
# [0, 1] one hot encoded: 1 -> red

print(f"data shape: {data.shape}")
print(f"labels shape: {labels.shape}")

class XOR(Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()               
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, hidden_dim)
        self.layer3 = Linear(hidden_dim, output_dim)
        
    def forward(self, inputs):
        # Forward pass
        
        out1 = self.layer1(inputs).tanh()
        out2 = self.layer2(out1).tanh()
        return self.layer3(out2)
    
model = XOR(2, 10, 1)

optimizer = Adam(model.parameters(), learning_rate=0.1)
criterion = BinaryCrossEntropyWithLogits()

EPOCHS = 100
history = []
for epoch in range(EPOCHS):
    # Gradients accumulates, therefore we need to set them to zero at each iteration
    optimizer.zero_grad()
    # Predictions
    predictions = model(data)
    loss = criterion(predictions, labels)
    # Compute the gradient
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Record the loss for plotting
    history.append(loss.data)
    if epoch % 10 == 0:
       print(f"epoch: {epoch} | loss: {loss.data:1.3E}")