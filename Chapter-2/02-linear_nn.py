import torch
from torch import nn, optim

w_true = torch.Tensor([1, 2, 3])

# Xのデータの準備
# 切片を回帰係数に含める（ax+b -> ax）ため、
# Xの最初の次元に1を追加
X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
print(X.size())

# 真の係数と各Xとの内積を行列とベクトルの積でまとめて計算
# with noise?
y = torch.mv(X, w_true) + torch.randn(100) * 0.5

# truth
a = torch.mv(X, w_true)

# Linear
# bias = False
net = nn.Linear(in_features=3, out_features=1, bias=False)

# initialize optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1)

# MSE loss (Mean Squared Error)
loss_fun = nn.MSELoss()

losses = []

# 100epoch
for epoch in range(100):
    # delete previously calculated grad
    optimizer.zero_grad()

    # prediction with linear model
    y_pred = net(X)

    # calculate MSE loss
    # y_pred's shape : (n,1) -> (n,)
    loss = loss_fun(y_pred.view_as(y), y)

    # calculate grad
    loss.backward()

    # update the gradient
    optimizer.step()

    # save losses to confirm the convergence
    losses.append(loss.item())

print(list(net.parameters()))
