import torch
import numpy as np
from time import time

# from torch.version import cuda
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

scaler = torch.tensor(10)
print(scaler)

vector = torch.tensor([1, 2, 3, -1, 5.3])
print(vector)

# create matrix by nested list
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)

matrix_1 = torch.tensor([[1, 2, 3, 3],
                         [4, 5, 6, 3],
                         [7, 8, 9, 3],
                         [10, 11, 12, 3]])

print(matrix_1)

# create column vector 2d
vector_col = torch.tensor([[5],
                           [3],
                           [6],
                           [9]])
print(vector_col)

# tensor 3D
tensor_3d = torch.tensor([[[1, 2],
                           [2, 3]],
                          [[4, 5],
                           [6, 7]]
                          ])
print(tensor_3d)

print(scaler.dtype)
# output :torch.int64
print(vector.dtype)
# output: torch.float32
print(matrix.dtype)
# output:torch.int64


sample_x = torch.tensor([1, 2, 3], dtype=torch.float64)
print(sample_x)  # output :tensor([1., 2., 3.], dtype=torch.float64)

sample_y = torch.tensor([1, -2, 3], dtype=torch.uint8)
print(sample_y)  # output NOTE!!! = tensor([  1, 254,   3], dtype=torch.uint8)

vector_uint = torch.tensor([1, 2, 3], dtype=torch.uint8)
vector_uint = vector_uint.half()
print(vector_uint)  # tensor([1., 2., 3.], dtype=torch.float16)
vector_uint = vector_uint.double()
print(vector_uint)  # tensor([1., 1., 1.], dtype=torch.float64)

vector_uint = vector_uint.type(torch.float64)
print(vector_uint)  # tensor([1., 2., 3.], dtype=torch.float64)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# avg_time = []
# matrix_2 = torch.randn(3000, 3000)
# # matric_2 = matrix_2.cuda()
# matrix_2 = matrix_2.cpu()
# for i in range(100):
#     start = time()
#     output = matrix_2.matmul(matrix_2)
#     avg_time.append(time() - start)  # 0.20432689428329467 On GPU || 0.24642916679382323 On CPU
#
# print(np.mean(avg_time))


print(vector.__class__)  # <class 'torch.Tensor'>
print(matrix_1.shape)  # torch.Size([4, 4])
print(matrix_1.size())  # torch.Size([4, 4])
print(matrix.min())  # tensor(1)
print(matrix.max())  # tensor(9)
print(matrix.float().mean())  # tensor(5.)
print(matrix.float().std())  # tensor(2.7386)
print(matrix.float().var())  # tensor(7.5000)

# transpose matrix
print(matrix.t())
# tensor([[1, 4, 7],
# [2, 5, 8],
# [3, 6, 9]])

tensor_ones = torch.ones((2, 2), dtype=torch.float64)
print(tensor_ones)

tensor_zeros = torch.zeros((3, 3, 2), dtype=torch.uint8)
print(tensor_zeros)

tensor_eye = torch.eye(8, 9, dtype=torch.uint8)
print(tensor_eye)

print(torch.full((5, 2, 3), 9.5))

# create tensor with the same size as  col_vec
print(torch.zeros_like(matrix))

# create tensor fill rand [0,1]
print(torch.rand(4, 5))
# create tensor fill rand normal distribution
print(torch.randn(4, 5))

import matplotlib.pyplot as plt

# Normal distribution
plt.hist(torch.rand(10000))
# plt.show()

# Gaussian distribution
plt.hist(torch.randn(10000), 100)
# plt.show()

print(torch.randint(20, 30, (3, 3)))

print(torch.randperm(20))

torch.manual_seed(123)
print(torch.randperm(20))

# Indexing AND Slicing

# Indexing
a = [1, 2, 3, 4, 5]
print(a[-1])
print(a[2])

b = torch.randint(1, 10, (3, 3))
print(b)
print(b[0, 0])
print(b[1, 2])
print(b[2])
# tensor([[2, 8, 2],
#         [8, 1, 8],
#         [1, 4, 1]])
# tensor(2)
# tensor(8)
# tensor([1, 4, 1])

tensor_3d = torch.randint(100, (3, 2, 3))
print(tensor_3d)

print(tensor_3d[0, 0, 0])

print(tensor_3d[1])

# Slicing
var_a = torch.rand(8)

print(var_a)
# tensor([0.4220, 0.5786, 0.9455, 0.8057, 0.6775, 0.6087, 0.6179, 0.6932])
print(var_a[0:1])
# tensor([0.4220])
print(var_a[2:5])
# tensor([0.9455, 0.8057, 0.6775])
print(var_a[6:])
# tensor([0.6179, 0.6932])
print(var_a[0:-2])
# tensor([0.4220, 0.5786, 0.9455, 0.8057, 0.6775, 0.6087])

print(var_a[::2])
# tensor([0.4220, 0.9455, 0.6775, 0.6179])

print(var_a[2::3])
# tensor([0.9455, 0.6087])


print(var_a[[0, 3, 5, 6]])
# tensor([0.4220, 0.8057, 0.6087, 0.6179])


var_b = torch.randn(3, 4)
print(var_b)
# tensor([[-1.2157,  0.1295,  0.0967,  1.4086],
#         [ 0.1915,  1.0041,  0.4198,  0.1882],
#         [-0.6040, -0.0417, -0.7580, -1.2113]])

print(var_b[0:2, 1:-1])
# tensor([[0.1295, 0.0967],
#         [1.0041, 0.4198]])

print(var_b[0:2, :])
# tensor([[-1.2157,  0.1295,  0.0967,  1.4086],
#         [ 0.1915,  1.0041,  0.4198,  0.1882]])

print(var_b[::2, 2])
# tensor([ 0.0967, -0.7580])


# Math Operations
mat_a = torch.randint(1, 6, (2, 2))
mat_b = torch.randint(1, 6, (2, 2))
print(mat_a)
# tensor([[3, 5],
#         [4, 4]]) 

print(mat_b)
# tensor([[3, 2],
#         [2, 2]])

print(mat_a + mat_b)
# tensor([[6, 7],
#         [6, 6]])
print(mat_a - mat_b)
# tensor([[0, 3],
#         [2, 2]])


print(mat_a * mat_b)
# tensor([[ 9, 10],
#         [ 8,  8]])

print(mat_a / mat_b)
# tensor([[1.0000, 2.5000],
#         [2.0000, 2.0000]])

print(torch.matmul(mat_a, mat_b))
# tensor([[19, 16],
#         [20, 16]])

# Broadcasting
b_cast = torch.randint(1, 6, (2, 2, 3))
print(b_cast)
# tensor([[[3, 3, 4],
#          [4, 3, 4]],
# 
#         [[4, 4, 1],
#          [2, 5, 4]]])
print(b_cast + 2)
# tensor([[[5, 5, 6],
#          [6, 5, 6]],
# 
#         [[6, 6, 3],
#          [4, 7, 6]]])

aa = torch.ones(3, 3)
bb = torch.arange(0, 3)
print(aa + bb)

# Dimension Transformation
a_1 = torch.randn(2, 6)
b_1 = torch.randn(3)

print(a_1)

print(a_1.view(2, 3, 2))

print(a_1.reshape(2, 3, 2))

# Transform tensor to vector
print(a_1.reshape(-1))
print(a_1.view(-1))

r = torch.randn(1, 6)
print(r)
# tensor([[-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533]])
print(r.repeat(4, 1))
# tensor([[-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533],
#         [-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533],
#         [-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533],
#         [-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533]])

print(torch.cat((r, r)))
# tensor([[-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533],
#         [-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533]])

print(torch.cat((r, r), dim=1))
# tensor([[-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533, -0.4979, -0.1584,
#           0.1426,  1.6831,  0.3073, -0.1533]])

# add new dim
print(r.shape)
# torch.Size([1, 6])
print(r.unsqueeze(2))
# tensor([[[-0.4979, -0.1584,  0.1426,  1.6831,  0.3073, -0.1533]]])
print(r.unsqueeze(2).shape)
# torch.Size([1, 6, 1])

# remove dim size 1
print(r.squeeze(-1).shape)
# torch.Size([1, 6])
