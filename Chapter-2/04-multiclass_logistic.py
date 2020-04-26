from sklearn.datasets import load_digits
import torch
from torch import nn, optim

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digits = load_digits()

X = digits.data
y = digits.target

X = torch.tensor(X, dtype=torch.float32)
# crossEntropyLoss function get int64 tensor
y = torch.tensor(y, dtype=torch.int64)

# output is 10-dim
net = nn.Linear(X.size()[1], 10)

# Choice device
X = X.to(device)
y = y.to(device)
net = net.to(device)

# Softmax cross entropy
loss_fn = nn.CrossEntropyLoss()

# SCD
optimizer = optim.SGD(net.parameters(), lr=0.01)

losses = []

# 100epoch
for epoch in range(100):
    # delete previously calculated grad
    optimizer.zero_grad()

    # calculate pred
    y_pred = net(X)

    # calculate grad
    loss = loss_fn(y_pred, y)
    loss.backward()

    # update gradiate
    optimizer.step()

    # save losses
    losses.append(loss.item())

# accuracy
# position and max are got from torch.max
_, y_pred = torch.max(net(X), 1)

# calculate accuracy
print((y_pred == y).sum().item() / len(y))
