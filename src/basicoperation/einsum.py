import torch

a = torch.randn((3,4))
b = torch.randn((4,3))
c1 = torch.matmul(a, b)
c2 = torch.einsum("ab, bc -> c", a, b)
print(c1.sum(dim=0), "\n", c2)











