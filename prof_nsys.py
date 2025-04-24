# %%
import torch

torch.ops.load_library("onesweep_cuda.cpython-310-x86_64-linux-gnu.so")

arr = torch.randint(0, 1 << 29, (1000000,), dtype=torch.uint32, device="cuda")

res = torch.ops.onesweep_cuda.one_sweep_sort(arr)

torch.sort(arr.to(torch.int32))
