import torch


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
print(gd_3) #tensor(1., requires_grad=True)
