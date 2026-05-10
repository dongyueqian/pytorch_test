import torch
'''
torch.tensor：从用户提供的已知数据创建张量。
torch.randn：根据指定形状创建随机初始化数据的张量。
torch.empty：创建未初始化数据的张量（只有空间结构，数据可能是垃圾值）。
'''

# # 标量scalar  0维
# scalar = torch.tensor(10)
# print(scalar.data) # 存储张量的实际数据，tensor(10)
# print(scalar.dtype) # 张量的数据类型 (如 torch.float32, torch.int64 等)，输出torch.int64
# print(scalar.device) # 张量的设备类型 (如 cpu, cuda:0, cuda:1 等)，输出cpu
# print(scalar.shape) # 张量的形状/维度，也返回torch.Size对象，torch.Size([])
# print(scalar.size()) # 张量的形状/维度，torch.Size([])
# print(scalar.grad)  # 张量的梯度 默认为 None
# print(scalar.item())  # 输出：10（获取标量的Python数值）

# # 向量 1维
# vec = torch.tensor([1,2,0])
# print(vec.shape)
# print(vec.data)
# print(vec.dtype)
# print(vec.grad)
#
# # 矩阵 2维
# mat = torch.tensor([[1,2,4],[2,3,7]]) # 2X3的矩阵
# print(mat.shape) #torch.Size([2, 3])
# print(mat.dtype) # torch.int64
# a = torch.tensor([[]])
# print(a.shape) # a是一个空的二维张量，torch.Size([1, 0])，这是一个1行0列的二维矩阵，属于合法的空张量。

# # 高维 3维
# tensor_3d = torch.randn(7,3,4)
# print(tensor_3d.shape) #torch.Size([7, 3, 4])
# print(tensor_3d.dtype) # torch.float32
# print(tensor_3d) #查看存储张量的实际数据

# tensor_4d = torch.randn(7,3,4,9)
# print(tensor_4d.shape) #torch.Size([7, 3, 4,9])
# print(tensor_4d.dtype) # torch.float32
# # print(tensor_4d) #查看存储张量的实际数据
# x_tensor_4d = tensor_4d.view(-1) #把4维展开成1维
# print(x_tensor_4d)


tensor = torch.tensor([[1,2,3], [2,3,1]])
tensor_T = tensor.T
print("Transposed Tensor:\n", tensor_T)

tensor2 = torch.tensor([2,2,5])
sum_tensor = torch.sum(tensor2)
print(sum_tensor)