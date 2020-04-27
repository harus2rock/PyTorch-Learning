import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import load_digits

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

digits = load_digits()

X = digits.data
Y = digits.target

# convert ndarray to tensor
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

# make dataset
ds = TensorDataset(X, Y)

# make DataLoader that returns 64 pieces of data in different order
loader = DataLoader(ds, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

# Choice device
X = X.to(device)
Y = Y.to(device)
net = net.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

losses = []
for epoch in range(10):
    running_loss = 0.0
    for xx, yy in loader:
        # xx and yy's size is 64
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)

# accuracy
_, y_pred = torch.max(net(X), 1)
Y = torch.tensor(Y, dtype=torch.int64)

# calculate accuracy
print((y_pred == Y).sum().item() / len(Y))
