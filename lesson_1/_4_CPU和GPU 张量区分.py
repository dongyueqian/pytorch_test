import torch

cpu_tensor = torch.tensor([2,3]).cpu()
# gpu_tensor = torch.randn(3,3).cuda()
print(cpu_tensor.device)
# print(gpu_tensor.device)