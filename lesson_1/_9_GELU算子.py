
'''GELU(Gaussian Error Linear Unit) 激活函数详解'''
# GELU 是一种先进的激活函数，被广泛应用于 Transformer 架构（如 GPT、BERT 等）中，
# 相比传统的 ReLU 和 LeakyReLU 有更好的性能表现。
# GELU 将神经元的输入与标准正态分布的累积分布函数(CDF)相乘
# gelu = torch.nn.GELU()
# x = torch.randn(2,3)
# out = gelu(x)
# print(out.shape)
# # 执行结果：
# # torch.Size([2, 3])
# # 作用：大模型主流激活函数。