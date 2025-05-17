import torch
import torch.nn.functional as F
from torch.linalg import matmul
from torch import optim


def funcx1(x):
    return x ** 2


def gradient_descent(func, xi, xeta, N):
    for iteration in range(N):
        func(xi).backward()
        xi.data -= eta * xi.grad
        xi.grad.zero_()
    # return xi.data
    return xi


xi = torch.tensor(1., requires_grad=True)
eta = 0.1
N = 50
gd_1 = gradient_descent(funcx1, xi, eta, N)
print(gd_1)  # tensor(1.4272e-05, requires_grad=True)


def funcx2(x):
    return torch.log(1 + torch.abs(x)) ** (2 + torch.sin(x))


xi = torch.tensor(1., requires_grad=True)
eta = 0.1
N = 50
gd_2 = gradient_descent(funcx2, xi, eta, N)
print(gd_2)  # tensor(0.0013, requires_grad=True)


def funcx3(x):
    return ((x ** 2) - 1) ** 3


xi = torch.tensor(1., requires_grad=True)
eta = 0.1
N = 30

gd_3 = gradient_descent(funcx3, xi, eta, N)
print(gd_3)  # tensor(1., requires_grad=True)


# create gradient_descent 3D 2 variable
def funcx4(x, y):
    return x * torch.exp(-x ** 2 - y ** 2) + 0.05 * (x ** 2 + y ** 2)


def gradient_descent3d(func, xi, yi, eta, N):
    for iteration in range(N):
        func(xi, yi).backward()
        xi.data -= eta * xi.grad
        yi.data -= eta * yi.grad
        xi.grad.zero_()
        yi.grad.zero_()
    # return xi.data
    return xi, yi


xi = torch.tensor(-2., requires_grad=True)
yi = torch.tensor(2., requires_grad=True)
eta = 0.1
N = 100

gd_4 = gradient_descent3d(funcx4, xi, yi, yi, N)
print(gd_4)  # (tensor(-0.6690, requires_grad=True), tensor(0.0103, requires_grad=True))

# connect Autograd to Neural Network
x = torch.tensor([[1., 2., 0., 4., 1.],
                  [0., 1., 1., 3., 2.],
                  [3., 2., 0., 5., 2.5]])
yt = torch.tensor([1., 2., 0.])


class Neuron:
    def __init__(self, m, af):
        self.w = torch.randn(m, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        self.af = af

    def __call__(self, x):
        if self.af == 'step':
            y = self.step(matmul(self.w, x) + self.b)
        elif self.af == 'linear':
            y = self.linear(matmul(self.w, x) + self.b)
        return y

    # create by step activation functions
    def step(self, x):
        if x > 0:
            y = torch.tensor([1.])
        elif x < 0:
            y = torch.tensor([0.])
        else:
            y = torch.tensor([0.5])
        return y

    def linear(self, x):
        return x


neuron_1 = Neuron(5, 'linear')
yp = neuron_1(x[0])
print(yt[[0]].shape)
print(yp.shape)
e = F.mse_loss(yp, yt[[0]])
print(e)  # tensor(21.1735)

print(e.backward())  # tensor(8.5489, grad_fn=<MseLossBackward0>)
print(neuron_1.w.grad)  # tensor([ -7.7594, -15.5188,  -0.0000, -31.0376,  -7.7594])
print(neuron_1.b.grad)  # tensor([-0.2774])


# Use optimizer pytorch

def funcx5(x):
    return x ** 2


xi = torch.tensor(-1., requires_grad=True)

params = [xi]
eta = 0.1
optimizer = optim.SGD(params, eta)
print(optimizer)
# Parameter Group 0
#     dampening: 0
#     differentiable: False
#     foreach: None
#     fused: None
#     lr: 0.1
#     maximize: False
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )

N = 25
for iteration in range(N):
    funcx5(xi).backward()
    optimizer.step()
    optimizer.zero_grad()

print(xi)  # tensor(-0.0038, requires_grad=True)


def funcx6(x,y):
    return x * torch.exp(-x ** 2 - y ** 2) + 0.05 * (x ** 2 + y ** 2)


xi = torch.tensor(-1.5, requires_grad=True)
yi = torch.tensor(1.5, requires_grad=True)

params = [xi,yi]
eta = 0.1
optimizer = optim.SGD(params, eta)

n = 100
for iteration in range(n):
    funcx6(xi,yi).backward()
    optimizer.step()
    optimizer.zero_grad()

print(xi)#tensor(-0.6691, requires_grad=True)
print(yi)#tensor(0.0006, requires_grad=True)

op= optimizer.param_groups[0]['params']
print(op)#[tensor(-0.6691, requires_grad=True), tensor(0.0006, requires_grad=True)]
