import torch

x1 = torch.tensor([[1, 2], [3, 4]])  # 2*2
x2 = torch.tensor([[10, 20, 30], [40, 50, 60]])  # 2*3

print(x1)
print(x2)

print('view 4*1 of 2*2')
print(x1.view(4, 1))

print('-1 is the rest of dimentions')
print(x1.view(1, -1))

print('transpose')
print(x2.t())

print('concatinate')
print(torch.cat([x1, x2], dim=1))

print('convert HWC to CHW')
hwc_img_data = torch.rand(100, 64, 32, 3)
chw_img_data = hwc_img_data.transpose(1, 2).transpose(1, 3)
print(hwc_img_data.size())
print(chw_img_data.size())
