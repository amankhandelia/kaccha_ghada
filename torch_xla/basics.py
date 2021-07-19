import torch
import torch_xla.core.xla_model as xm

# assignment
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(xm.xla_device())
print(x)

# operation
y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(y.device)
print(x+y)

x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(xm.xla_device())
y = torch.tensor([[2,3,5], [7, 8, 9], [12, 54, 7]], dtype=torch.float32).to(xm.xla_device())
z = torch.cat([x, y], dim=1)
print(z.shape)
print(z)

# broadcasting

x = torch.arange(2).reshape(1, 2).to(xm.xla_device())
y = torch.arange(3).reshape(3, 1).to(xm.xla_device())
print(x+y)

# broadcasting

x = torch.arange(20).reshape(2, 5, 2).to(xm.xla_device())
y = torch.arange(5).reshape(5, 1).to(xm.xla_device())
print(x+y)

