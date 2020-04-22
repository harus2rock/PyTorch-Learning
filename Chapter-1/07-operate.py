import torch
import time

m = torch.randn(100, 10)
v = torch.randn(10)
print(m.size())
print(v.size())

print('internal product')
d = torch.dot(v, v)
print(d)

print('product')
v2 = torch.mv(m, v)
print(v2.size())

print('matrix product')
m2 = torch.mm(m.t(), m)
print(m2.size())

start = time.time()
print('singular value decomposition')
u, s, v = torch.svd(m)
end = time.time()
print('elapsed_time:{0}'.format(end - start) + '[sec]')
print('with cpu: 0.00026488304138183594[sec]')
