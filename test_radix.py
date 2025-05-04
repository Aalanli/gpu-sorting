# %%

import torch
import utils
from functools import partial

torch.ops.load_library("onesweep2.cpython-310-x86_64-linux-gnu.so")
# torch.ops.load_library("onesweep_cuda.cpython-310-x86_64-linux-gnu.so")

def radix_sort(arr: torch.Tensor, kernel: int = 0):
    return torch.ops.onesweep2.onesweep_b32(arr, kernel)

def num_kernels():
    return torch.ops.onesweep2.num_kernels()

kernel_policy = {
    (0, 4096): 4,
    (4096, 16384): 3, 
    (16384, 131072): 2,
    (262144, 300000): 1,
    (300000, int(1e11)): 0
}

def radix_sort_auto(arr: torch.Tensor):
    N = arr.shape[0]
    for (low, high), kernel in kernel_policy.items():
        if low <= N < high:
            return radix_sort(arr, kernel)
    return radix_sort(arr, 0)

mk_args = lambda size: (torch.randint(0, 1 << 29, (size,), dtype=torch.int32, device="cuda"),)

def bench():
    functions = [
        ("torch", lambda arr: torch.sort(arr)[0]),
        ("onesweep_auto", radix_sort_auto),
    ] + [
        ("onesweep", partial(radix_sort, kernel=i)) for i in range(num_kernels())
    ]

    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], [524288, 1048576, 2500000, 4000000, 6000000]
    for s in sizes:
        res = utils.benchmark_algos(functions, mk_args, s, warmup=10, runs=200)
        utils.pretty_print_res(res)
        utils.plot_benchmark_results(res)
        print("")

def test():
    sizes = [1, 512, 16087]
    for s in sizes:
        arr = torch.randint(0, 1 << 29, (s,), dtype=torch.int32, device="cuda")
        arr2 = torch.sort(arr)[0]
        for kernel in range(0, num_kernels()):
            arr1 = radix_sort(arr, kernel)
            assert (arr1 == arr2).all(), f"Mismatch at size {s} and kernel {kernel}: {arr1} != {arr2}"

def nsys():
    # ncu --target-processes all --set full -f -o radix_sort python test_radix.py
    for s in [391000]:
        arr = torch.randint(0, 1 << 29, (s,), dtype=torch.int32, device="cuda")
        radix_sort(arr)
        torch.sort(arr)
if utils.is_under_ncu():
    print("Running with ncu")
    nsys()
else:
    test()
    bench()
