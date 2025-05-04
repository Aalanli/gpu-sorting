#include <torch/script.h>
#include "coalesced_sweep.h"

#include <iostream>


torch::Tensor compute_histograms_b32(
    torch::Tensor data, bool hist_init, int64_t nreps_per_thread, int64_t threads, int64_t radix_log) {
    // Check input tensor
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kInt32, "data must be of type uint32");
    TORCH_CHECK(data.dim() == 1, "data must be a 1D tensor");
    const int N = data.size(0);
    
    // Allocate output tensor
    int hist_rows = (32 + radix_log - 1) / radix_log;
    int hist_cols = 1 << radix_log;
    torch::Tensor hist;
    if (hist_init) {
        hist = torch::empty({hist_rows, hist_cols}, torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA));        
    } else {
        hist = torch::zeros({hist_rows, hist_cols}, torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA));        
    }

    // Launch the kernel
    launch_compute_histograms_b32(
        reinterpret_cast<const uint32_t*>(data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(hist.data_ptr<int>()),
        N,
        hist_init,
        nreps_per_thread,
        threads,
        radix_log
    );

    return hist;
}

torch::Tensor compute_histograms_pipelined_b32(
    torch::Tensor data, bool hist_init, int64_t nreps_per_thread, int64_t npipe) {
    // Check input tensor
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kInt32, "data must be of type uint32");
    TORCH_CHECK(data.dim() == 1, "data must be a 1D tensor");
    const int N = data.size(0);
    
    // Allocate output tensor
    int hist_rows = 4;
    int hist_cols = 1 << 8;
    torch::Tensor hist;
    if (hist_init) {
        hist = torch::empty({hist_rows, hist_cols}, torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA));        
    } else {
        hist = torch::zeros({hist_rows, hist_cols}, torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA));        
    }

    // Launch the kernel
    launch_compute_histograms_pipelined_b32(
        reinterpret_cast<const uint32_t*>(data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(hist.data_ptr<int>()),
        N,
        hist_init,
        nreps_per_thread,
        npipe
    );

    return hist;
}

torch::Tensor compute_histograms_shared_b32(
    torch::Tensor data, int64_t nreps_per_thread, int64_t threads, int64_t radix_dup
) {
    // Check input tensor
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kInt32, "data must be of type uint32");
    TORCH_CHECK(data.dim() == 1, "data must be a 1D tensor");
    const int N = data.size(0);
    
    // Allocate output tensor
    int hist_rows = 4;
    int hist_cols = 1 << 8;
    torch::Tensor hist = torch::zeros({hist_rows, hist_cols}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA));        

    // Launch the kernel
    launch_compute_histograms_shared_b32(
        reinterpret_cast<const uint32_t*>(data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(hist.data_ptr<int>()),
        N,
        hist_cols,
        nreps_per_thread,
        threads,
        radix_dup
    );

    return hist;
}

torch::Tensor compute_histograms_shared_pipelined_b32(
    torch::Tensor data, int64_t threads, int64_t nreps_per_thread, int64_t radix_dup, int64_t npipe
) {
    // Check input tensor
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kInt32, "data must be of type uint32");
    TORCH_CHECK(data.dim() == 1, "data must be a 1D tensor");
    const int N = data.size(0);
    
    // Allocate output tensor
    int hist_rows = 4;
    int hist_cols = 1 << 8;
    torch::Tensor hist = torch::zeros({hist_rows, hist_cols}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA));        

    // Launch the kernel
    launch_compute_histograms_shared_pipelined_b32(
        reinterpret_cast<const uint32_t*>(data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(hist.data_ptr<int>()),
        N,
        hist_cols,
        threads,
        nreps_per_thread,
        radix_dup,
        npipe
    );

    return hist;
}

torch::Tensor single_radix_pass_b32(torch::Tensor key) {
    // Check input tensors
    TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(key.scalar_type() == torch::kInt32, "key must be of type uint32");
    TORCH_CHECK(key.dim() == 1, "key must be a 1D tensor");

    const int N = key.size(0);

    auto global_hist = compute_histograms_pipelined_b32(
        key, false, 16, 2
    ); // [4, 256]

    

    auto res = torch::zeros_like(key, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA));
    
    const int elem_per_block = 256 * 15;
    const int blocks = (N + elem_per_block - 1) / elem_per_block;
    auto local_hist = torch::zeros({blocks + 1, 256}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA));
    auto index = torch::zeros({1}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA));
    
    launch_single_radix_pass_b32(
        reinterpret_cast<const uint32_t*>(key.data_ptr<int>()),
        reinterpret_cast<const uint32_t*>(global_hist.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(res.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(local_hist.data_ptr<int>()),
        N,
        0,
        index.data_ptr<int>()
    );

    return res;
}

TORCH_LIBRARY(sweep, m) {
    m.def("compute_histograms_b32(Tensor data, bool hist_init, int nreps_per_thread, int threads, int radix_log) -> Tensor", compute_histograms_b32);
    m.def("compute_histograms_pipelined_b32(Tensor data, bool hist_init, int nreps_per_thread, int npipe) -> Tensor", compute_histograms_pipelined_b32);
    m.def("compute_histograms_shared_b32(Tensor data, int nreps_per_thread, int threads, int radix_dup) -> Tensor", compute_histograms_shared_b32);
    m.def("compute_histograms_shared_pipelined_b32(Tensor data, int threads, int nreps_per_thread, int radix_dup, int npipe) -> Tensor", compute_histograms_shared_pipelined_b32);
    m.def("single_radix_pass_b32(Tensor key) -> Tensor", single_radix_pass_b32);
}
