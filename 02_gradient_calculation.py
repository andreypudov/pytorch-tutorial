import torch

x = torch.rand(3, requires_grad = True)
print("x = ", x)

y = x + 2
print("y = ", y)

z = y * y * 2
print("z = ", z)

z = z.mean()
print("z = ", z)

# without z.mean()
# v = torch.tensor([0.1, 1.0, 0.01], dtype = torch.float32)
# z.backward(v)
z.backward() # dz / dx
print("x = ", x.grad)

# x.required_grad(False)
# x.detach()
# with torch.no_grad()

# x.requires_grad_(False)
# print("x = ", x)

# y = x.detach()
# print("y = ", y)

# with torch.no_grad():
#     y = x + 2
#     print("y = ", y)

weights = torch.ones(4, requires_grad = True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print("weights = ", weights.grad)

    weights.grad.zero_()

# optimizer = torch.optim.SGD(weights, lr = 0.01)
# optimizer.step()
# optimizer.zero_grad()
