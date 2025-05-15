import torch
from sympy.physics.biomechanics import activation

# sample input
x = torch.tensor([[1., 2., 0., 4., 1.],
                  [0., 1., 1., 1., 1.],
                  [2., 3., 0., 1., 4.]])

w = torch.tensor([1., 0.5, 1., -1., -0.5])
b = torch.tensor(1.)


# create linear activation function
def linear(x):
    return x


# create step activation function
def step(x):
    if x > 0:
        y = torch.tensor([1.])

    elif x < 0:
        y = torch.tensor([0.])
    else:
        y = torch.tensor([0.5])
    return y


# create Neuron
def neuron(x, w, b, af):
    z = 0
    for xi, wi in zip(x, w):
        z += xi * wi
    z += b
    y = af(z)
    return y


# test Neuron
# print(neuron(x[0], w, b, linear))  # tensor(-1.5000)

print(neuron(x[0], w, b, step))  # tensor([0.])
