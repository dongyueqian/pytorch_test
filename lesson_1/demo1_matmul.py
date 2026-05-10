import torch
import pytest
import time

'''
算子测试的核心维度：
1、数值正确性
对比手工计算结果：assert torch.allclose(actual, expected)
相关PyTorch工具：torch.allclose

2、边界条件
空张量/极大值/极小值：torch.tensor([])
相关PyTorch工具：
- torch.finfo：看浮点型（float16/float32/float64）的信息
- torch.iinfo：看整数型（int8/int32/int64）的信息

3、数据类型
测试不同dtype的兼容性：float16, int8等
相关PyTorch工具：tensor.to(dtype=...)

4、性能基准
耗时和内存占用：time.perf_counter()
相关PyTorch工具：torch.cuda.synchronize()

5、异常处理
非法输入时的行为：传入非张量数据
相关PyTorch工具：pytest.raises(...)
'''
# 点乘函数
def func(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
# 测试代码
def test_dot_product_accuracy():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 6.0, 1.0])
    r = func(a , b)
    c = torch.dot(a,b) # 官方实现
    print(f"差异：{abs(r - c)}")
    assert torch.allclose(torch.dot(a,b), r)

def test_dot_product_performance():
    a2 = torch.randn(100000)
    b2 = torch.randn(100000)
    # 测试时间
    start = time.perf_counter()
    torch.dot(a2, b2)
    # func(a2, b2) #超过10毫秒，用例失败
    end = time.perf_counter()
    elapsd = (end - start) * 1000
    print(f"执行时长：{elapsd} 毫秒")
    assert elapsd < 10.0