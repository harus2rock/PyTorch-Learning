import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

digits = load_digits()

X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# convert ndarray to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

# more deep neural netwark
k = 100
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# make dataset
ds = TensorDataset(X_train, Y_train)

# make DataLoader that returns 32 pieces of data in different order
loader = DataLoader(ds, batch_size=32, shuffle=True)

# Choice device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)
net = net.to(device)

train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    # change the network mode to train
    net.train()
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    # change the network mode to validation
    net.eval()
    y_pred = net(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())

plt.plot(train_losses)
plt.plot(test_losses)
plt.show()

# accuracy
_, y_pred = torch.max(net(X_test), 1)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

# calculate accuracy
print((y_pred == Y_test).sum().item() / len(Y_test))
