#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
#include "one_sweep2.cuh"
#include <tuple>
#include <vector>

const static std::vector<AbstractLauncher<uint32_t>* > kernels = {
    new OneSweepLauncher<uint32_t, 32, 8, 512, 13>(),
    new OneSweepLauncher<uint32_t, 32, 8, 256, 9>(),
    new OneSweepLauncher<uint32_t, 32, 8, 256, 7>(),
    new OneSweepLauncher<uint32_t, 32, 8, 256, 3>(),
    new OneSweepLauncher<uint32_t, 32, 8, 256, 1>(),
};


torch::Tensor onesweep_b32(
    torch::Tensor data, int64_t kernel
) {
    TORCH_CHECK(data.device().is_cuda(), "data must be a CUDA tensor");
    TORCH_CHECK(data.scalar_type() == torch::kInt32, "data must be of type uint32");
    TORCH_CHECK(data.dim() == 1, "data must be a 1D tensor");
    const int N = data.size(0);

    TORCH_CHECK(0 <= kernel && kernel < kernels.size(), "kernel must be in range [0, ", kernels.size(), ")");
    auto onesweep = kernels[kernel];
    int workspace_size = onesweep->get_workspace_size_in_bytes(N);

    auto workspace = torch::empty({workspace_size}, torch::TensorOptions()
        .dtype(torch::kBool)
        .device(data.device()));
    
    auto res = torch::empty_like(data);

    onesweep->launch(
        reinterpret_cast<const uint32_t*>(data.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(res.data_ptr<int>()),
        reinterpret_cast<uint8_t*>(workspace.data_ptr<bool>()),
        N, 
        c10::cuda::getCurrentCUDAStream().stream()
    );

    return res;
}

int64_t num_kernels() {
    return kernels.size();
}

TORCH_LIBRARY(onesweep2, m) {
    m.def("onesweep_b32(Tensor data, int kernel) -> Tensor", &onesweep_b32);
    m.def("num_kernels() -> int", &num_kernels);   
    // m.def("digit_bins_b32(Tensor data) -> Tensor", &digit_bins_b32);
}

