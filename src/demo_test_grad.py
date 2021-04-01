import torch

#x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[1, 2], [3, 4]], dtype=float, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)
print(y.grad)
print(z.grad)