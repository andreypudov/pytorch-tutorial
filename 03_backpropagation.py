import torch

# x = 1
# y = 2
# w = 1
#
# x       ^
#  \      y        s   ^
#   [*]------[-]------[s]------[loss]
#  /         /
# y         y
#
# FORWARD
# ^                   ^
# y =  w * x, loss = (y - y)^2 = (wx - y)^2
#
# ^
# y = 1 * 1 = 1, s = 1 - 2 = -1, loss = (-1)^2 = 1
#
#                          ^           ^
# dloss   ds^2       ds   dy - y      dy   dwx
# ----- = ---- = 2s, -- = ------ = 1, -- = --- = x
#  ds      ds         ^      ^        dw    dw
#                    dy     dy
#
# BACKWARD
#                                                       ^
# dloss   dloss   ds                   dloss   dloss   dy
# ----- = ----- * -- = 2 * s * 1 = -2, ----- = ----- * -- = -2 * x = -2
#   ^       ds     ^                     dw       ^    dw
#  dy             dy                             dy

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad = True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2
print("loss = ", loss)

# backward pass
loss.backward()
print("grad = ", w.grad)

## update weights
## next forward and backward
