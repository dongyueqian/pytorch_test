import time
import pytest
import torch

'''测试框架：基础测试'''
class TestDotProduct:
    def test_accuracy(self):
        # 准确度测试
        a = torch.tensor([1.0, 2.0]) # 1维张量
        b = torch.tensor([3.0, 4.0]) # 1维张量
        assert torch.dot(a, b) == 11, "点积计算错误"

    def test_performance(self):
        a = torch.randn(1000)
        b = torch.randn(1000)
        start = time.time()
        torch.dot(a, b)
        assert (time.time() - start) < 0.01, "计算超时"

    @pytest.mark.parametrize("x,y,expected",[
        ([1, 2], [3, 4], 11),
        ([0, 0], [0, 0], 0),
        ([1e10, 2e10], [3e10, 4e10], 11e20)
    ])
    def test_dot_parametrized(self,x, y, expected):
        a = torch.tensor(x)
        b = torch.tensor(y)
        res = torch.dot(a, b)
        assert torch.allclose(res ,torch.tensor(expected)),f"预期结果：{expected},实际结果：{res}"