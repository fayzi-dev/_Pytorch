import torch

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
                         [7, 8, 9, 3]])
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
