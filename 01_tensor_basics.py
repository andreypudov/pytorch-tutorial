import torch
import numpy as np

x = torch.ones(2, 2, dtype=torch.int)
x = torch.tensor([2.5, 0.1])
# print(x, x.dtype, x.size())

x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print("x = ", x)
# print("y = ", y)
z = x + y
# z = torch.add(x, y)
# y.add_(x)
# sub, mul, div
# print("z = ", z)

x = torch.rand(5, 5)
# print("x = ", x)
# print("x[:, 0] = ", x[:, 0]) # column zero
# print("x[0, :] = ", x[0, :]) # row zero
# print("x[1, 1] = ", x[1, 1], x[1, 1].item())

x = torch.rand(4, 4)
# print("x = ", x)
# y = x.view(16)
y = x.view(-1, 8)
#print("y = ", y)

a = torch.ones(5)
# print("a = ", a, type(a))
b = a.numpy()
# print("b = ", b, type(b))
a.add_(1)
# print("a = ", a)
# print("b = ", b)

a = np.ones(5)
# print("a = ", a)
b = torch.from_numpy(a)
# print("b = ", b)
a += 1
# print("a = ", a)
# print("b = ", b)

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy() # error
    z = z.to("cpu")
    #print("x = ", x)

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy() # error
    z = z.to("cpu")
    # print("x = ", x)

    x = torch.ones(5, requires_grad =  True)
    # print("x = ", x)
