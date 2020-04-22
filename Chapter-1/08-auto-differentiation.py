import torch

x = torch.randn(100, 3)
print('change the requires_grad flag to true if use differentiation')
a = torch.tensor([1, 2, 3.], requires_grad=True)

print('operate')
y = torch.mv(x, a)
o = y.sum()

print('differentiation')
o.backward()

print('compare')
print(a.grad != x.sum(0))
print(a.grad)
print(x.sum(0))
print(x.grad is None)
