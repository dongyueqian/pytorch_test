import torch

# 尺寸边界：1×1极小张量
a = torch.tensor([[1]])
b = torch.tensor([[2]])
print(a.shape, b.shape)
c = a @ b
print(c.shape,c.dtype)
print(c.itemsize)

'''
数值边界
在 PyTorch 的矩阵乘法（@ 或 torch.matmul）中，
数值边界指的是输入或输出张量中可能出现的极端数值情况，
这些情况可能导致计算异常、精度损失或数值不稳定。
以下是常见的数值边界情况及处理方法
算子边界：int16:（-32768, 32767）
float16:
'''
# # 1、数值溢出 Overflow ，当计算结果超出数据类型（dtype）的表示范围时发生。
# # 常见于 float16（半精度）或 int 类型，

# int16 数值溢出演示
# val = torch.tensor(32768, dtype=torch.int16)
# print(val)
# 执行结果（溢出异常）：
# tensor(-32768, dtype=torch.int16)

# a1 = torch.tensor([[1e3]], dtype=torch.float16)
# b1 = torch.tensor([[1e3]], dtype= torch.float16)
# c1 = a1 @ b1
# print(c1) # tensor([[inf]], dtype=torch.float16)

#
# # 2、数值下溢（Underflow）
# # 当数值接近零，超出数据类型的有效精度范围时，被截断为零。
# # 常见于 float16 或小数值乘法。
# a2 = torch.tensor([[1e-4]], dtype= torch.float16)
# b2 = torch.tensor([[1e-4]], dtype= torch.float16)
# c2 = a2 @ b2
# print(c2) # 结果 0（实际应为 1e-8，但 float16 无法表示）
#
# a22 = torch.tensor([[1e-30]], dtype= torch.float32)
# b22 = torch.tensor([[1e-30]], dtype= torch.float32)
# c22 = a22 @ b22
# print(c22) # 结果 0（实际应为 1e-60，但 float32 无法表示）

# 3、非数值（NaN/Inf）[NaN:Not a Number, Inf:Infinity 无穷大]
# 非法运算（如 0/0、∞ - ∞）会产生 NaN。
# 溢出会产生 Inf。
a3 = torch.tensor([[0.0]])
b3 = torch.tensor([[0.0]])
c3 = a3 / b3  # 非法运算，结果 tensor([[nan]])
print(c3)

a4 = torch.tensor([[1e30]])
b4 = torch.tensor([[1e30]])
c4 = a4 @ b4  # 无穷大，可能输出 tensor([[inf]])
print(c4)
