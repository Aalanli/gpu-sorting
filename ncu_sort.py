import torch

torch.ops.load_library("onesweep_cuda.cpython-310-x86_64-linux-gnu.so")

for n in [391000, 4000000]:
    arr = torch.randint(0, 1 << 29, (n,), dtype=torch.uint32, device="cuda")

    res = torch.ops.onesweep_cuda.one_sweep_sort(arr)

    arr = arr.view(torch.int32)
    torch.sort(arr)
