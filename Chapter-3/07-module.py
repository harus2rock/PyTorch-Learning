import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class CustomLinear(nn.Module):
    def __init__(self, in_features,
                 out_features, bias=True, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class MyMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ln1 = CustomLinear(in_features, 200)
        self.ln2 = CustomLinear(200, 200)
        self.ln3 = CustomLinear(200, 200)
        self.ln4 = nn.Linear(200, out_features)

    def forward(self, x):
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        return x


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
mlp = MyMLP(64, 10)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters())

# make dataset
ds = TensorDataset(X_train, Y_train)

# make DataLoader that returns 32 pieces of data in different order
loader = DataLoader(ds, batch_size=32, shuffle=True)

# Choice device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)
mlp = mlp.to(device)

train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    # change the network mode to train
    mlp.train()
    for i, (xx, yy) in enumerate(loader):
        y_pred = mlp(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    # change the network mode to validation
    mlp.eval()
    y_pred = mlp(X_test)
    test_loss = loss_fn(y_pred, Y_test)
    test_losses.append(test_loss.item())

plt.plot(train_losses)
plt.plot(test_losses)
plt.show()

# accuracy
_, y_pred = torch.max(mlp(X_test), 1)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

# calculate accuracy
print((y_pred == Y_test).sum().item() / len(Y_test))
