# %%
import torch

torch.ops.load_library("onesweep_cuda.cpython-310-x86_64-linux-gnu.so")

arr = torch.randint(0, 1 << 29, (10000,), dtype=torch.uint32, device="cuda")

res = torch.ops.onesweep_cuda.one_sweep_sort(arr)

(torch.diff(res.to(torch.int32)) >= 0).all()

# %%
import time
import matplotlib.pyplot as plt
import numpy as np

def benchmark_sort_algorithm(data, sort_fn, warmup=3, runs=10):
    # Warmup runs
    for _ in range(warmup):
        sort_fn(data.clone())
    
    # Timed runs
    times = []
    for _ in range(runs):
        data_copy = data.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        sort_fn(data_copy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)

def benchmark_sorting(sizes, warmup=3, runs=10):
    torch_results = []
    onesweep_results = []
    
    for size in sizes:
        print(f"\nBenchmarking size {size:,}")
        # Generate random data
        data = torch.randint(0, 1000000, (size,), dtype=torch.uint32, device="cuda")
        
        # Benchmark torch.sort
        torch_mean, torch_std = benchmark_sort_algorithm(
            data.to(torch.int32), 
            lambda x: torch.sort(x),
            warmup=warmup,
            runs=runs
        )
        torch_results.append((torch_mean, torch_std))
        
        # Benchmark onesweep
        onesweep_mean, onesweep_std = benchmark_sort_algorithm(
            data,
            lambda x: torch.ops.onesweep_cuda.one_sweep_sort(x),
            warmup=warmup,
            runs=runs
        )
        onesweep_results.append((onesweep_mean, onesweep_std))

    return torch_results, onesweep_results

# Test with various sizes
sizes = list(range(1000, 400000, 5000)) #[50000, 100000, 500000, 1000000, 2500000, 5000000]
torch_results, onesweep_results = benchmark_sorting(sizes, warmup=10, runs=100)

# Extract means and stds for plotting
torch_means, torch_stds = zip(*torch_results)
onesweep_means, onesweep_stds = zip(*onesweep_results)

# Plot results
plt.figure(figsize=(12, 6))
plt.errorbar(sizes, torch_means, yerr=torch_stds, fmt='o-', label='torch.sort', capsize=5)
plt.errorbar(sizes, onesweep_means, yerr=onesweep_stds, fmt='o-', label='onesweep', capsize=5)
plt.xlabel('Array Size')
plt.ylabel('Time (ms)')
plt.title('Sorting Performance Comparison (with error bars)')
plt.legend()
plt.grid(True)
plt.show()

# Print detailed results
print("\nDetailed Results:")
print("Size\t\tTorch (ms)\tTorch Std\tOneSweep (ms)\tOneSweep Std")
print("-" * 70)
for i, size in enumerate(sizes):
    print(f"{size:,}\t{torch_means[i]:.2f}\t\t{torch_stds[i]:.2f}\t\t{onesweep_means[i]:.2f}\t\t{onesweep_stds[i]:.2f}")



