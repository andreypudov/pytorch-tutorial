# Activation Functions
# 1) Step Function
#           | 1, if x >= 0
#    f(x) = |
#           | 0 otherwise
#
# 2) Sigmoid
#               1
#    f(x) = --------
#           1 + e^-1
#
#    probability [0, 1]
#
# 3) TanH
#               1
#    f(x) = --------- - 1
#           1 + e^-2x
#
#    probability [-1, 1]
#
# 4) ReLU
#
#    f(x) = max(0, x)
#
# 5) Leaky ReLU
#           | x, if x >= 0
#    f(x) = |
#           | a * x otherwise
#
# 6) Softmax
#            e^yi
#    S(yi) = -----
#            Σe^yj
#
#    probability [0, 1]

import torch
import torch.nn as nn
# import torch.nn.functional as F

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # F.relu()
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
