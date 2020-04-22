# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch


# %%
# numpyメソッドを使用してndarrayに変換
t = torch.tensor([[1, 2], [3, 4.]])
x = t.numpy()


# %%
# GPU上のTensorはcpuメソッドで、一度CPUのTensorに変換する必要がある
t = torch.tensor([[1, 2], [3, 4]], device="cuda:0")
x = t.to("cpu").numpy()
