import torch
from torch import nn, optim
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# _/_/_/ Train

iris = load_iris()

# iris is 3 class classification progrem
# here, only use two classes
X = iris.data[:100]
y = iris.target[:100]

# convert ndarray to Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# iris data is 4-dim
net = nn.Linear(4, 1)

# calculate cross entropy with sigmoid function
loss_fn = nn.BCEWithLogitsLoss()

# SGD
optimizer = optim.SGD(net.parameters(), lr=0.25)

losses = []

# 100 epoch
for epoch in range(100):
    # delete previous calculated grad
    optimizer.zero_grad()

    # predict y with linear model
    y_pred = net(X)

    # calculate grad
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()

    # update the gradient
    optimizer.step()

    # save losses
    losses.append(loss.item())

plt.plot(losses)
plt.show()

# _/_/_/ Prediction
h = net(X)

# probability of y=1
prob = nn.functional.sigmoid(h)

# out is ByteTensor because pytorch doesn't have Bool type
y_pred = prob > 0.5

# check the result
print((y.byte() == y_pred.view_as(y)).sum().item())
