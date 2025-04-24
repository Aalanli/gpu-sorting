#include <torch/library.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <tuple>
#include "one_sweep.cuh"

class OneSweepDispatcher
{
    const uint32_t k_maxSize;
    const uint32_t k_radix = 256;
    const uint32_t k_radixPasses = 4;
    const uint32_t k_partitionSize = 7680;
    const uint32_t k_globalHistPartitionSize = 65536;
    const uint32_t k_globalHistThreads = 128;
    const uint32_t k_binningThreads = 512;
    const uint32_t k_valPartSize = 4096;
    

    uint32_t* m_index;
    uint32_t* m_globalHistogram;
    uint32_t* m_firstPassHistogram;
    uint32_t* m_secPassHistogram;
    uint32_t* m_thirdPassHistogram;
    uint32_t* m_fourthPassHistogram;
    uint32_t* m_errCount;

    uint32_t* m_alt = nullptr;
    uint32_t* m_altPayload = nullptr;
    uint32_t alt_size = 0;

public:
    OneSweepDispatcher(uint32_t maxSize) : k_maxSize(maxSize)
    {
        const uint32_t maxBinningThreadblocks = divRoundUp(k_maxSize, k_partitionSize);
        cudaMalloc(&m_index, k_radixPasses * sizeof(uint32_t));
        cudaMalloc(&m_globalHistogram, k_radixPasses * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_firstPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_secPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_thirdPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_fourthPassHistogram, maxBinningThreadblocks * k_radix * sizeof(uint32_t));
        cudaMalloc(&m_errCount, 1 * sizeof(uint32_t));
    }

    ~OneSweepDispatcher()
    {
        if (m_alt != nullptr)
        {
            cudaFree(m_alt);
            cudaFree(m_altPayload);
        }
        cudaFree(m_index);
        cudaFree(m_globalHistogram);
        cudaFree(m_firstPassHistogram);
        cudaFree(m_secPassHistogram);
        cudaFree(m_thirdPassHistogram);
        cudaFree(m_fourthPassHistogram);
        cudaFree(m_errCount);
    }

    void allocate_alt(uint32_t size)
    {
        if (m_alt == nullptr || size > alt_size)
        {
            cudaFree(m_alt);
            cudaFree(m_altPayload);
            cudaMalloc(&m_alt, size * sizeof(uint32_t));
            cudaMalloc(&m_altPayload, size * sizeof(uint32_t));
            alt_size = size;
        }
    }

    static inline uint32_t divRoundUp(uint32_t x, uint32_t y)
    { 
        return (x + y - 1) / y;
    }

    void ClearMemory(uint32_t binningThreadBlocks)
    {
        cudaMemset(m_index, 0, k_radixPasses * sizeof(uint32_t));
        cudaMemset(m_globalHistogram, 0, k_radix * k_radixPasses * sizeof(uint32_t));
        cudaMemset(m_firstPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_secPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_thirdPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
        cudaMemset(m_fourthPassHistogram, 0, k_radix * binningThreadBlocks * sizeof(uint32_t));
    }

    void DispatchKernelsKeysOnly(uint32_t size, uint32_t* sort, uint32_t* sortResult)
    {
        assert(size <= k_maxSize);
        allocate_alt(size);

        const uint32_t globalHistThreadBlocks = divRoundUp(size, k_globalHistPartitionSize);
        const uint32_t binningThreadBlocks = divRoundUp(size, k_partitionSize);

        ClearMemory(binningThreadBlocks);

        cudaDeviceSynchronize();

        OneSweep::GlobalHistogram <<<globalHistThreadBlocks, k_globalHistThreads >>>(sort, m_globalHistogram, size);

        OneSweep::Scan <<<k_radixPasses, k_radix >>> (m_globalHistogram, m_firstPassHistogram, m_secPassHistogram,
            m_thirdPassHistogram, m_fourthPassHistogram);

        OneSweep::DigitBinningPassKeysOnly <<<binningThreadBlocks, k_binningThreads >>> (sort, m_alt, m_firstPassHistogram,
            m_index, size, 0);

        OneSweep::DigitBinningPassKeysOnly <<<binningThreadBlocks, k_binningThreads >>> (m_alt, sortResult, m_secPassHistogram,
            m_index, size, 8);

        OneSweep::DigitBinningPassKeysOnly <<<binningThreadBlocks, k_binningThreads >>> (sortResult, m_alt, m_thirdPassHistogram,
            m_index, size, 16);

        OneSweep::DigitBinningPassKeysOnly <<<binningThreadBlocks, k_binningThreads >>> (m_alt, sortResult, m_fourthPassHistogram,
            m_index, size, 24);
    }

    void DispatchKernelsPairs(uint32_t size, uint32_t* sort, uint32_t* sortPayload, uint32_t* sortResult, uint32_t* sortResultPayload)
    {
        assert(size <= k_maxSize);
        allocate_alt(size);

        const uint32_t globalHistThreadBlocks = divRoundUp(size, k_globalHistPartitionSize);
        const uint32_t binningThreadBlocks = divRoundUp(size, k_partitionSize);

        ClearMemory(binningThreadBlocks);

        cudaDeviceSynchronize();

        OneSweep::GlobalHistogram <<<globalHistThreadBlocks, k_globalHistThreads >>>(sort, m_globalHistogram, size);

        OneSweep::Scan <<<k_radixPasses, k_radix >>> (m_globalHistogram, m_firstPassHistogram, m_secPassHistogram,
            m_thirdPassHistogram, m_fourthPassHistogram);

        OneSweep::DigitBinningPassPairs <<<binningThreadBlocks, k_binningThreads >>> (sort, sortPayload, m_alt, 
            m_altPayload, m_firstPassHistogram, m_index, size, 0);

        OneSweep::DigitBinningPassPairs <<<binningThreadBlocks, k_binningThreads >>> (m_alt, m_altPayload, sortResult,
            sortResultPayload, m_secPassHistogram, m_index, size, 8);

        OneSweep::DigitBinningPassPairs <<<binningThreadBlocks, k_binningThreads >>> (sortResult, sortResultPayload, m_alt,
            m_altPayload, m_thirdPassHistogram, m_index, size, 16);

        OneSweep::DigitBinningPassPairs <<<binningThreadBlocks, k_binningThreads >>> (m_alt, m_altPayload, sortResult,
            sortResultPayload, m_fourthPassHistogram, m_index, size, 24);
    }

    uint32_t get_max_size() {
        return k_maxSize;
    }

};

// Create a global instance of the sorter
static OneSweepDispatcher sorter(1000);

void resize_sorter(uint32_t size)
{
    sorter.~OneSweepDispatcher();
    new (&sorter) OneSweepDispatcher(size);
}

torch::Tensor one_sweep_sort(torch::Tensor input) {
    auto size = input.size(0);
    TORCH_CHECK(input.dtype() == torch::kUInt32, "Input must be of type uint32");
    TORCH_CHECK(input.dim() == 1, "Input must be a 1D tensor");
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be on a CUDA device");
    auto sortResult = torch::empty_like(input);
    if (size > sorter.get_max_size()) {
        resize_sorter(float(size * 1.5));
    }
    sorter.DispatchKernelsKeysOnly(size, input.data_ptr<uint32_t>(), sortResult.data_ptr<uint32_t>());
    return sortResult;
}

std::tuple<torch::Tensor, torch::Tensor> one_sweep_sort_pairs(torch::Tensor input, torch::Tensor payload) {
    auto size = input.size(0);
    TORCH_CHECK(input.dtype() == torch::kUInt32, "Input must be of type uint32");
    TORCH_CHECK(payload.dtype() == torch::kUInt32, "payload must be of type uint32");
    TORCH_CHECK(input.dim() == 1, "Input must be a 1D tensor");
    TORCH_CHECK(payload.dim() == 1, "payload must be a 1D tensor");
    TORCH_CHECK(input.device().type() == torch::kCUDA, "Input must be on a CUDA device");
    TORCH_CHECK(payload.device().type() == torch::kCUDA, "payload must be on a CUDA device");
    auto sortResult = torch::empty_like(input);
    auto sortResultPayload = torch::empty_like(payload);
    if (size > sorter.get_max_size()) {
        resize_sorter(float(size * 1.5));
    }
    sorter.DispatchKernelsPairs(size, input.data_ptr<uint32_t>(), payload.data_ptr<uint32_t>(), sortResult.data_ptr<uint32_t>(), sortResultPayload.data_ptr<uint32_t>());
    return std::make_tuple(sortResult, sortResultPayload);
}

// Register the operation with PyTorch
TORCH_LIBRARY(onesweep_cuda, m) {
    m.def("one_sweep_sort(Tensor input) -> Tensor", 
        [](const torch::Tensor& input) {
            return one_sweep_sort(input);
        });
    m.def("one_sweep_sort_pairs(Tensor input, Tensor payload) -> (Tensor, Tensor)", 
        [](const torch::Tensor& input, const torch::Tensor& payload) {
            return one_sweep_sort_pairs(input, payload);
        });
}