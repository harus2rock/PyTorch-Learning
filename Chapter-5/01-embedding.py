import torch
from torch import nn

# express 10000 kinds of token using 20-dim vector
emb = nn.Embedding(10000, 20, padding_idx=0)
# input is Tensor of int64
inp = torch.tensor([1, 2, 5, 2, 10], dtype=torch.int64)
# output is Tensor of float32
out = emb(inp)

print(out)
