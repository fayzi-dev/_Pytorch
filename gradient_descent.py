import torch


def funcx2(x):
    return x ** 2


def gradient_descent(func, xi, xeta, N):
    for iteration in range(N):
        func(xi).backward()
        xi.data -= eta * xi.grad
        xi.grad.zero_()
    # return xi.data
    return xi
xi =torch.tensor(1., requires_grad=True)
eta=0.1
N=50
G_D = gradient_descent(funcx2,xi,eta,N)
print(G_D) #tensor(1.4272e-05, requires_grad=True)