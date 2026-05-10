import torch

# # 基础矩阵
# matA = torch.randn(3,4)
# matB = torch.randn(4,9)
# matC = matA @ matB #或者
# m = torch.matmul(matA,matB)
# # 左列=右行才能乘
# print(matC.shape)
# print(m.shape)

# # 高维矩阵,第一维是批量数，后面两维做矩阵乘法，大模型Transformer层常用。
# # 批数相等，且batch_a的最后一维=batch_b的倒数第二维，才能做矩阵乘法
# batch_a = torch.randn(2,3,4)
# batch_b = torch.randn(2,4,5)
# batch_c = torch.matmul(batch_a,batch_b)
# print(batch_c.shape)

# batch_m和batch_n的批数不相等，但是batch_n的形状被隐试试为（1，7，5），批数=1，
# pytorch将batch_n沿批数纬度复制到(2,7,5)，以匹配batch_m的批数2
# 批数为1的矩阵可以广播到任何批数。
# 若两个张量的批数均不为1且不相等，则报错
batch_m = torch.randn(2,5,7)
batch_n = torch.randn(7,5)
print(torch.matmul(batch_m,batch_n).shape)
