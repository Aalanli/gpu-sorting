# %%
from typing import Callable
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def is_under_ncu():
    return "NV_TPS_LAUNCH_TOKEN" in os.environ


def benchmark(f, args, warmup=3, runs=10):
    # Warmup runs
    for _ in range(warmup):
        f(*args)
    
    # Timed runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        f(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    return times


def benchmark_algos(
    functions: list[tuple[str, Callable]],
    mk_args: Callable,
    sizes: list[int],
    warmup: int = 3,
    runs: int = 10,
):
    function_names = {}
    new_functions = []
    for n, f in functions:
        if n in function_names:
            new_functions.append((f"{n}_{function_names[n]}", f))
            function_names[n] += 1
        else:
            function_names[n] = 0
            new_functions.append((n, f))
    functions = new_functions
    results = {
        n: (sizes, []) for n, _ in functions
    }
    for size in sizes:
        args = mk_args(size)
        for name, f in functions:
            times = benchmark(f, args, warmup=warmup, runs=runs)
            results[name][1].append(times)
    return results

def plot_benchmark_results(results, title="Benchmark Results"):
    plt.figure(figsize=(10, 6))
    for name, (sizes, times) in results.items():
        mean_times = np.mean(times, axis=1)
        std_times = np.std(times, axis=1)
        plt.errorbar(sizes, mean_times, yerr=std_times, label=name)
    plt.xlabel('Size')
    plt.ylabel('Time (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def pretty_print_res(res):
    data = {}
    for name, (sizes, times) in res.items():
        mean_times = np.mean(times, axis=1)        
        data[name] = mean_times
    
    df = pd.DataFrame(data, index=sizes).T
    df.index.name = "Size"
    for col in df.columns:
        min_val = df[col].min()
        df[col] = df[col].apply(lambda x: f"*{x:.4f}*" if x == min_val else f"{x:.4f}")
    print(df)


