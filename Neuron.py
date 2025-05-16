# Create Neuron By Class
from operator import matmul

import torch
from torch import nn
import torch.nn.functional as F


class Neuron:
    def __init__(self, m, af):
        self.w = torch.randn(m)
        self.b = torch.randn(1)
        self.af = af

    def __call__(self, x):
        y = self.af(matmul(self.w, x) + self.b)
        return y


# create linear activation function
def linear(x):
    return x


neuron = Neuron(5, linear)
# print(neuron) #<__main__.Neuron object at 0x0000028370D1CB80>
# print(neuron.w) #tensor([ 0.6657, -1.1392,  0.1168])
# print(neuron.b) #tensor([1.7826])


inputs = torch.tensor([[1., 2., 0., 4., 1.],
                       [0., 1., 1., 1., 1.],
                       [2., 3., 0., 1., 4.]])

# print(neuron(inputs[0]))  # tensor([6.1333])


# Loss Functions

# data for prediction & target
yp = torch.tensor([-0.30, -0.40, -0.50, 0.65, 0.55, -0.32, 0.87, -0.44, 0.75, 1.33])
yt = torch.tensor([-0.31, -0.20, -0.20, 0.55, 0.50, -0.28, 0.79, -0.34, 0.70, 1.5])

# MSE
# From Scratch
mse_1 = torch.mean((yp - yt) ** 2)
# print(mse_1)  # tensor(0.0192)

# From class  MSE by pytorch
mse_2 = nn.MSELoss()
# print(mse_2(yp, yt))  # tensor(0.0192)

# From function  MSE by pytorch
mse_3 = F.mse_loss(yp, yt)
# print(mse_3)  # tensor(0.0192)

# MAE
# From Scratch
mae_1 = torch.mean(torch.abs(yp - yt))
# print(mae_1)  # tensor(0.1100)

# From class  MAE by pytorch
mae_2 = nn.L1Loss()
# print(mae_2(yp, yt))  # tensor(0.1100)

# From function  MAE by pytorch
mae_3 = F.l1_loss(yp, yt)
# print(mae_3)  # tensor(0.1100)


# Connect Neuron to loss function
class NeuronNetwork:
    def __init__(self, m, af):
        self.w = torch.randn(m)
        self.b = torch.randn(1)
        self.af = af

    def __call__(self, x):
        if self.af == 'linear':
            y = self.linear(matmul(self.w, x) + self.b)
        return y

    def linear(self, x):
        return x


x = torch.tensor([[1., 2., 0., 4., 1.],
                  [0., 1., 1., 1., 1.],
                  [2., 3., 0., 1., 4.]])
# print(x[0].shape)
yt = torch.tensor([0.01, 0.03, 0.5, 0.8, 1.])

neuron_1 = NeuronNetwork(5, 'linear')

yp = neuron_1(x[0])
# print(yp.shape)

# e = F.mse_loss(yp, yt)
# print(e) # tensor(24.9681)
# ERROR !!!!!! =Neuron.py:98: UserWarning: Using a target size (torch.Size([5])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#   e = F.mse_loss(yp, yt)

# e = F.mse_loss(yp, yt[0])
# print(yt[0].shape) #torch.Size([])
# print(yp.shape) #torch.Size([1])
# print(e) #Neuron.py:103: UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  # e = F.mse_loss(yp, yt[0])

e =F.mse_loss(yp, yt[[0]])
print(yt[[0]].shape) #torch.Size([1])
print(yp.shape)#torch.Size([1])
print(e) #tensor(2.2142)

