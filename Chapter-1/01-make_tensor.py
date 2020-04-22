# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch


# %%
# 入れ子のlistを渡して作成
t = torch.tensor([[1, 2], [3, 4]])


# %%
# deviceを指定することでGPUにTensorを作成する
t = torch.tensor([[1, 2], [3, 4]], device='cuda:0')


# %%
# dtypeを指定することで倍精度のTensorを作る
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)


# %%
# 0から9までの数値で初期化された1次元のTensor
t = torch.arange(0, 10)


# %%
# すべての値が0の100×10のTensorを作成し、toメソッドでGPUに転送する
t = torch.zeros(100, 10).to('cuda:0')


# %%
# 正規乱数で100×10のTensorを作成
t = torch.randn(100, 10)


# %%
# Tensorのshapeはsizeメソッドで取得可能
t.size()
