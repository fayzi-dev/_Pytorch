import torch
import time

N = 1000

A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu


print(f"CPU Time: {cpu_time:.6f} seconds")

if torch.cuda.is_available():

    A_gpu = A_cpu.to("cuda")
    B_gpu = B_cpu.to("cuda")

    _ = torch.mm(A_gpu, B_gpu)

    torch.cuda.synchronize()
    start_gpu = time.time()
    C_gpu = torch.mm(A_gpu, B_gpu)
    torch.cuda.synchronize()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"GPU Time: {gpu_time:.6f} seconds")
else:
    print("CUDA is not available on this system.")

# CPU Time: 0.021944 seconds
# GPU Time: 0.004231 seconds