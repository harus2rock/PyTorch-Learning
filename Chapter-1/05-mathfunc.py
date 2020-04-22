import torch

print('100×10のテストデータを用意')
X = torch.randn(100, 10)
print(X)

print('数学関数を含めた数式')
y = X * 2 + torch.abs(X)
print(y)

print('平均値')
m = torch.mean(X)
print(m)

print('method')
m = X.mean()
print(m)

print('get value with item method')
m_value = m.item()
print(m_value)

print('specify the dimension')
m2 = X.mean(0)
print(m2)
