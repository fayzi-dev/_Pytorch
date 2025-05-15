# Create Neuron By Class
from operator import matmul

import torch


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

print(neuron(inputs[0]))  # tensor([6.1333])
