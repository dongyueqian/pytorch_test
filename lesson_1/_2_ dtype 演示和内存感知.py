import torch

# 整型
i16 = torch.tensor(10000, dtype=torch.int16)
i32 = torch.tensor(9990000, dtype=torch.int32)
print(i16.data,i16.shape,i16.grad)
print(i16.dtype, i32.dtype)
print(i16.dtype.itemsize, i32.dtype.itemsize) # 输出2代表占有2个字节的内存，4代表占有4个字节的内存

# 浮点型
f16 = torch.tensor(100, dtype=torch.float16) # 半精度
f32 = torch.tensor(100, dtype=torch.float32) # 单精度
f64 = torch.tensor(100, dtype=torch.float64) # 双精度
print(f16.dtype, f32.dtype, f64.dtype)
print(f16.dtype.itemsize, f32.dtype.itemsize, f64.dtype.itemsize)
# 输出2代表占有2个字节的内存，4代表占有4个字节的内存，8代表占有8个字节的内存
# 布尔型
b = torch.tensor(True, dtype=torch.bool)
print(b.dtype)
print(b.dtype.itemsize) # 1代表占有1个字节的内存