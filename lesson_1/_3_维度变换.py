import torch

x = torch.randn(2,6,8) # 创建一个三维张量x
print(x.shape)
x1 = x.reshape(3,4,8) #
print(x1.shape)
x2 = x.view(24,4)
print(x2.shape)