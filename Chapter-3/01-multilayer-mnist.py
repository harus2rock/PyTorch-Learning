import torch
from torch import nn, optim
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

digits = load_digits()

X = digits.data
Y = digits.target

# convert ndarray to tensor
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

# Choice device
X = X.to(device)
Y = Y.to(device)
net = net.to(device)

# Softmax cross entropy
loss_fn = nn.CrossEntropyLoss()

# Adam
optimizer = optim.Adam(net.parameters())

losses = []

# 100epoch
for epoch in range(500):
    # delete previously calculated grad
    optimizer.zero_grad()

    # prediction
    y_pred = net(X)

    # calculate grad
    loss = loss_fn(y_pred, Y)
    loss.backward()

    # update grad
    optimizer.step()

    # save loss
    losses.append(loss.item())

# accuracy
_, y_pred = torch.max(net(X), 1)
Y = torch.tensor(Y, dtype=torch.int64)

# calculate accuracy
print((y_pred == Y).sum().item() / len(Y))

# confusion matrix
plt.figure()
sns.heatmap(confusion_matrix(Y, y_pred), annot=True)
# plt.show()
