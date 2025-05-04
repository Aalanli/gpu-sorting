# %%
import torch
import utils

torch.ops.load_library("sweep.cpython-310-x86_64-linux-gnu.so")

def compute_hist_sweep_custom(
    arr: torch.Tensor,
    init_hist: bool = False,
    thread_reps: int = 10,
    radix_log: int = 8,
    threads: int = 256,
):
    return torch.ops.sweep.compute_histograms_b32(
        arr,
        init_hist,
        thread_reps,
        threads,
        radix_log,
    )

def compute_hist_sweep_custom_pipelined(
    arr: torch.Tensor,
    init_hist: bool = False,
    thread_reps: int = 10,
    npipeline: int = 2
):
    return torch.ops.sweep.compute_histograms_pipelined_b32(
        arr,
        init_hist,
        thread_reps,
        npipeline,
    )


def compute_hist_sweep_custom_shared(
    arr: torch.Tensor,
    thread_reps: int = 10,
    threads: int = 256,
    radix_dup: int = 2
):
    return torch.ops.sweep.compute_histograms_shared_b32(
        arr,
        thread_reps,
        threads,
        radix_dup,
    )


def compute_hist_sweep_custom_shared_pipelined(
    arr: torch.Tensor,
    threads: int = 256,
    thread_reps: int = 10,
    radix_dup: int = 2,
    npipeline: int = 2
):
    return torch.ops.sweep.compute_histograms_shared_pipelined_b32(
        arr,
        threads,
        thread_reps,
        radix_dup,
        npipeline,
    )

def compute_bins(arr, h, radix_log):
    radix = 1 << radix_log
    return ((arr >> (h * radix_log)) & (radix - 1)).bincount(minlength=radix)

def compute_hist_sweep(arr, radix_log):
    hists = []
    for h in range(4):
        hist = compute_bins(arr, h, radix_log)
        hists.append(hist)
    hists = torch.stack(hists)
    return torch.cumsum(hists, dim=1) - hists[:, 0:1]
    # return torch.stack(hists)


mk_args = lambda size: (torch.randint(0, 1 << 29, (size,), dtype=torch.int32, device="cuda"),)

def test():
    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 4000000]

    for size in sizes:
        args = mk_args(size)
        res1 = compute_hist_sweep(*args, 8)
        torch.cuda.synchronize()
        res2 = compute_hist_sweep_custom(*args, False, 10, 8, 128)
        torch.cuda.synchronize()
        res3 = compute_hist_sweep_custom_pipelined(*args, False, 10, 2)
        torch.cuda.synchronize()
        res4 = compute_hist_sweep_custom_pipelined(*args, True, 10, 2)
        torch.cuda.synchronize()
        res5 = compute_hist_sweep_custom_shared(*args)
        torch.cuda.synchronize()
        res6 = compute_hist_sweep_custom_shared_pipelined(*args)
        torch.cuda.synchronize()

        # print(res1[0] == res2[0])
        assert (res1 == res2).all(), f"Mismatch at size {size}: {res1} != {res2}"
        assert (res1 == res3).all(), f"Mismatch at size {size}: {res1} != {res3}"
        assert (res1 == res4).all(), f"Mismatch at size {size}: {res1} != {res4}"
        assert (res1 == res5).all(), f"Mismatch at size {size}: {res1} != {res5}"
        assert (res1 == res6).all(), f"Mismatch at size {size}: {res1} != {res6}"
        print(f"Test passed for size {size}")

# test()
# torch.random.manual_seed(42)
# arr = mk_args(1024 * 4)[0]
# # arr = torch.arange(128, device="cuda", dtype=torch.int32)
# res1 = compute_hist_sweep(arr, 8)
# res2 = compute_hist_sweep_custom(arr, False, 10, 8)
# print(res1[0])
# print(res2[0])
# print(res1[0] == res2[0])
# print((res1 == res2).all())

def bench():
    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2500000, 4000000, 6000000]
    functions = [
        # ("torch", lambda arr: compute_hist_sweep(arr, 8)),
        # ("sweep_init", lambda arr: compute_hist_sweep_custom(arr, True, 16, 8, 128)),
        # ("sweep_init_2", lambda arr: compute_hist_sweep_custom(arr, True, 16, 8, 256)),
        # ("sweep_no_init", lambda arr: compute_hist_sweep_custom(arr, False, 10, 8, 64)),
        # ("sweep_no_init", lambda arr: compute_hist_sweep_custom(arr, False, 10, 8, 128)),
        ("sweep_no_init", lambda arr: compute_hist_sweep_custom(arr, False, 10, 8, 256)),
        ("sweep_no_init", lambda arr: compute_hist_sweep_custom(arr, False, 10, 8, 512)),

        # ("sweep_init_pipelined", lambda arr: compute_hist_sweep_custom_pipelined(arr, True, 10, 2)),
        ("sweep_no_init_pipelined_2", lambda arr: compute_hist_sweep_custom_pipelined(arr, False, 16, 2)),
        # ("sweep_no_init_pipelined_3", lambda arr: compute_hist_sweep_custom_pipelined(arr, False, 16, 3)),
        # ("sweep_shared", lambda arr: compute_hist_sweep_custom_shared(arr, 10, 256, 1)),
        # ("sweep_shared", lambda arr: compute_hist_sweep_custom_shared(arr, 10, 256, 2)),
        ("sweep_shared", lambda arr: compute_hist_sweep_custom_shared(arr, 10, 512, 1)),
        # ("sweep_shared", lambda arr: compute_hist_sweep_custom_shared(arr, 10, 512, 2)),

        ("sweep_shared_pipelined", lambda arr: compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 1, 2)),
        ("sweep_shared_pipelined", lambda arr: compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 2, 2)),
        ("sweep_shared_pipelined", lambda arr: compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 1, 3)),
        # ("sweep_shared_pipelined", lambda arr: compute_hist_sweep_custom_shared_pipelined(arr, 512, 16, 2, 2)),
    ]

    res = utils.benchmark_algos(functions, mk_args, sizes, warmup=10, runs=200)
    utils.pretty_print_res(res)
    utils.plot_benchmark_results(res)


def nsys():
    # ncu --target-processes all --set full -f -o sweep python test_coalesced_sweep.py  
    for s in [391000, 4000000]:
        arr = torch.randint(0, 1 << 29, (s,), dtype=torch.int32, device="cuda")
        # compute_hist_sweep_custom(arr, True, 10, 8, 256)
        # compute_hist_sweep_custom(arr, False, 10, 8, 256)
        # compute_hist_sweep_custom_pipelined(arr, False, 16, 2)
        # compute_hist_sweep_custom_pipelined(arr, True, 16, 2)
        # compute_hist_sweep_custom_shared(arr, 10, 256, 1)
        # compute_hist_sweep_custom_shared(arr, 10, 256, 2)
        compute_hist_sweep_custom_shared(arr, 10, 512, 1)
        # compute_hist_sweep_custom_shared(arr, 10, 512, 2)
        compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 1, 2)
        compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 2, 2)
        compute_hist_sweep_custom_shared_pipelined(arr, 512, 24, 1, 3)

if utils.is_under_ncu():
    print("Running with ncu")
    nsys()
else:
    bench()
    # test()

# %%
