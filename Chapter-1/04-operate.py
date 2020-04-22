import torch

print('長さ3のベクトル')
v = torch.tensor([1, 2, 3.])
w = torch.tensor([0, 10, 20.])
print(v)
print(w)

print('2×3の行列')
m = torch.tensor([[0, 1, 2], [100, 200, 300.]])
print(m)

print('ベクトルとスカラーの足し算')
v2 = v + 10
print(v2)

print('累乗も同様')
v2 = v ** 2
print(v2)

print('同じ長さのベクトル同士の引き算')
z = v - w
print(z)

print('複数の組み合わせ')
u = 2 * v - w / 10 + 6.0
print(u)

print('行列とスカラー')
m2 = m * 2.0
print(m2)

print('行列とベクトル')
m3 = m + v
print(m3)

print('行列同士')
m4 = m + m
print(m4)
