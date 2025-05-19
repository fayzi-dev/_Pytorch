import torch
from torch import nn, device

fully_connected = nn.Linear(in_features=5, out_features=1)
print(fully_connected.bias)  # tensor([-0.2728], requires_grad=True)
print(fully_connected.weight)  # tensor([[ 0.3011,  0.3618,  0.0141, -0.2982, -0.3797]], requires_grad=True)

x = torch.randn(10, 5)
y = fully_connected(x)
print(y)  # tensor([[-0.0330],
# [-0.5174],
# [-1.3506],
# [-0.1104],
# [ 0.2714],
# [-0.5059],
# [-0.3185],
# [ 0.8094],
# [ 0.4880],
# [-1.1869]], grad_fn=<AddmmBackward0>)


fully_connected_1 = nn.Linear(in_features=4, out_features=5, bias=True)
print(fully_connected_1.bias)
print(fully_connected_1.weight)
x = torch.randint(low=1, high=100, size=(6, 4))
print(x)
print(x.shape)



x_2 = torch.randint(low=1, high=100, size=(9, 5), dtype=torch.float)
fully_connected_2 = nn.Linear(in_features=5, out_features=10, bias=True)
fully_connected_3= nn.Linear(in_features=10, out_features=5, bias=True)

print(fully_connected_3(fully_connected_2(x_2)))
# tensor([[ -9.7780,   2.0473,  22.0571,   0.6000,  -4.4704],
#         [-16.4642, -24.0825,  47.4492, -10.9426,  -4.9306],
#         [-18.2518,  -7.6804,  38.7734,  -7.0680,  -1.3442],
#         [-22.6804,  -5.0801,  31.0128,  -4.3901,   0.1922],
#         [-21.5313, -11.1448,  31.8221,  -2.9398,  -3.4093],
#         [-11.2720, -16.1190,  30.3660,  -6.0820,  -3.6959],
#         [-15.5479,   4.4269,  22.3217,  -1.0640,   2.9326],
#         [ -7.7021, -18.5538,  24.9916,  -8.7759,  -0.6070],
#         [-19.2550, -14.6038,  37.8844, -10.7503,   3.8537]],
#        grad_fn=<AddmmBackward0>)
mlp = nn.Sequential(nn.Linear(in_features=5, out_features=10),
                    nn.Linear(in_features=10, out_features=5))

print(mlp)
#Sequential(
#   (0): Linear(in_features=5, out_features=10, bias=True)
#   (1): Linear(in_features=10, out_features=5, bias=True)
# )

print(mlp(x_2))
# tensor([[  7.1508,  15.8593,  13.4327, -35.4424, -10.2979],
#         [ -6.8449,  22.7472,  30.0235, -65.2610,   2.4617],
#         [ 21.2462,   9.4576,  11.8946, -65.4565,   8.9218],
#         [ 19.0841,  19.4358,   4.0513, -47.0014,  14.9276],
#         [-11.0343,  43.0002,  21.2856, -36.2412,  -1.5716],
#         [-13.8171,  24.2445,  24.6888, -39.3235,  -4.4060],
#         [ -8.0059,  37.6373,  25.7976, -50.0577, -20.5378],
#         [ -5.3496,   6.0877,  14.4847, -35.1464,   8.2083],
#         [ -1.6320,  24.1873,  24.0330, -68.5691,   3.8303]],
#        grad_fn=<AddmmBackward0>)


