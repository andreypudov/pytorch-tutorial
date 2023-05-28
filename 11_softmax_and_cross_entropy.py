# Softmax
#
#        e^yi
# S(yi) =-----
#        Σe^yi
#
# output to [0, 1]

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis = 0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f'softmax numpy: {outputs}')

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim = 0)
print(f'softmax numpy: {outputs}')

# Cross-Entropy Loss
#
#   ^        1             ^
# D(Y,Y) = - - * ΣYi * log(Yi)
#            N
#
# Y = [1, 0, 0]            ^
# ^                   => D(Y,Y) = 0.35
# Y = [0.7, 0.2, 0.1]
#
# Y = [1, 0, 0]            ^
# ^                   => D(Y,Y) = 2.3
# Y = [0.1, 0.3, 0.6]
#
# the better prediction, then low the loss

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss
# must not implement softmax in last layer
# Y has class labels (not One-Hot)
# Y_pred has raw scores (logits)
loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1])
# nsamples * nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 numpy: {l1.item():.4f}')
print(f'Loss2 numpy: {l2.item():.4f}')
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Predictions1 numpy: {predictions1}')
print(f'Predictions2 numpy: {predictions2}')

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size = 28 * 28, hidden_size = 5, num_classes = 3)
criterion = nn.CrossEntropyLoss() # (applies softmax)

# Binary:
# sigmoid + nn.BCELoss()

# Binary classification problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet2(input_size = 28 * 28, hidden_size = 5)
criterion = nn.BCELoss()
