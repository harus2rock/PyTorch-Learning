import torch
from matplotlib import pyplot as plt

# 真の係数
w_true = torch.Tensor([1, 2, 3])

# Xのデータの準備
# 切片を回帰係数に含める（ax+b -> ax）ため、
# Xの最初の次元に1を追加
X = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
print(X)
print(X.size())

# 真の係数と各Xとの内積を行列とベクトルの積でまとめて計算
# with noise?
y = torch.mv(X, w_true) + torch.randn(100) * 0.5
print(y)

# truth
a = torch.mv(X, w_true)
print(a)

# 勾配降下で最適化するためのパラメータのTensor
w = torch.randn(3, requires_grad=True)

# 学習率
gamma = 0.1

# 損失関数のログ
losses = []

# 100epoch
for epoch in range(100):
    # 前回のbackward method で計算された勾配の値を削除
    w.grad = None

    # 線形モデルでのyの予測値を計算
    y_pred = torch.mv(X, w)

    # MSE lossとwによる微分を計算
    # truth
    loss = torch.mean((a - y_pred)**2)
    # with noise
    loss = torch.mean((y - y_pred)**2)
    loss.backward()

    # 勾配更新
    # wをそのまま代入して更新：計算グラフ破壊（異なるTensorになる）
    # dataのみ更新
    w.data = w.data - gamma * w.grad.data

    # 収束確認のためにlossを記録しておく
    losses.append(loss.item())

plt.plot(losses)
# plt.show()

print(w)
