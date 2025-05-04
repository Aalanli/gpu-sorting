#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <unordered_map>

namespace cg = cooperative_groups;

//General macros
#define LANE_COUNT          32							//Threads in a warp
#define LANE_MASK           31							//Mask of the lane count
#define LANE_LOG            5							//log2(LANE_COUNT)

static constexpr int FLAG_NOT_READY = 0;
static constexpr int FLAG_REDUCTION = 1;
static constexpr int FLAG_INCLUSIVE = 2;
static constexpr int FLAG_MASK = 3;


__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }
    return val;
}


__device__ __forceinline__ uint32_t InclusiveActiveWarpScan(uint32_t val) {
    const uint32_t mask = __activemask();
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(mask, val, i, 32);
        if (getLaneId() >= i) val += t;
    }
    return val;
}


__device__ __forceinline__ uint32_t ExclusiveWarpScan(uint32_t val) {
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) // 16 = LANE_COUNT >> 1
    {
        const uint32_t t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    const uint32_t t = __shfl_up_sync(0xffffffff, val, 1, 32);
    return getLaneId() ? t : 0;
}

/*
1. compute global histogram and overlap computation with computation of local histogram
sync_grid();
2. lookback and move into sort payload
*/

struct WLMS {
    uint32_t warp_flags; // bit mask indicating which thread has the same bit pattern as the current thread
    uint32_t bits; // the number of lanes lesser than the current one which has the same bit pattern
};

template <int NBits, typename T>
__device__ __forceinline__ WLMS wlms(T key) {
    unsigned warp_flags = 0xffffffff;

    #pragma unroll
    for (int k = 0; k < NBits; ++k) {
        const bool t2 = key >> k & 1;
        warp_flags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
    }
    const uint32_t bits = __popc(warp_flags & getLaneMaskLt());
    return WLMS { warp_flags, bits };
}

template <int NBits, typename T>
__device__ __forceinline__ WLMS wlmsMasked(T key, bool mask) {
    unsigned warp_flags = __ballot_sync(0xffffffff, mask);;
    #pragma unroll
    for (int k = 0; k < NBits; ++k) {
        const bool t2 = key >> k & 1;
        warp_flags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
    }
    const uint32_t bits = __popc(warp_flags & getLaneMaskLt());
    return WLMS { warp_flags, bits };
}

__host__ __device__ __forceinline__ int divup(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T, int HISTOGRAMS, int RADIX_LOG, int THREADS, bool INIT_GLOBAL>
__global__ void __launch_bounds__(THREADS) compute_histograms(
    const T *__restrict__ data, // [N]
    uint32_t *__restrict__ g_hists, // [HISTOGRAMS, RADIX]
    const int N
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    static_assert(RADIX_LOG <= 16 && RADIX_LOG >= 2, "");

    __shared__ uint32_t s_hists[HISTOGRAMS * RADIX];
    if (INIT_GLOBAL) {
        for (int i = threadIdx.x + blockIdx.x * THREADS; i < HISTOGRAMS * RADIX; i += THREADS * gridDim.x) {
            g_hists[i] = 0;
        }
    }
    for (int i = threadIdx.x; i < HISTOGRAMS * RADIX; i += THREADS) {
        s_hists[i] = 0;
    }
    __syncthreads();
    for (int i = 0; i < divup(N, THREADS * gridDim.x); i++) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        T dreg = (offset < N) ? data[offset] : 0;
        for (int h = 0; h < HISTOGRAMS; ++h) {
            uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
            WLMS d_wlms = wlmsMasked<RADIX_LOG>(dh, offset < N);
            auto cnt = __popc(d_wlms.warp_flags);
            if (d_wlms.bits == 0 && offset < N && cnt > 0) {
                atomicAdd(s_hists + h * RADIX + dh, cnt);
            }
        }
    }
    __syncthreads();
    if (INIT_GLOBAL) {
        auto grid = cooperative_groups::this_grid();
        grid.sync();
    }


    // use a warp to do scan for each histogram for now
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    for (int i = warp_id; i < HISTOGRAMS; i += THREADS / 32) {
        uint32_t s_accum = 0;
        uint32_t s_first = s_hists[i * RADIX];
        
        for (int j = lane_id; j < RADIX; j += 32) {
            const uint32_t s = InclusiveWarpScan(s_hists[i * RADIX + j]);
            atomicAdd(g_hists + i * RADIX + j, s + s_accum - s_first);
            s_accum += __shfl_sync(0xffffffff, s, 31);
        }
    }
}

thread_local std::unordered_map<void*, int> kernel_blocks;

template <typename F>
int get_kernel_blocks(F kernel, int blocks, int threads) {
    auto kernel_addr = reinterpret_cast<void*>(&kernel);
    if (kernel_blocks.count(kernel_addr) == 0) {
        int numBlocksPerSm = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, threads, 0);
        blocks = min(numBlocksPerSm * deviceProp.multiProcessorCount, blocks);
        kernel_blocks[kernel_addr] = numBlocksPerSm * deviceProp.multiProcessorCount;
    } else {
        blocks = min(kernel_blocks[kernel_addr], blocks);
    }
    return blocks;
}

// TODO pass stream parameter
template <typename T, int NBITS, int RADIX_LOG, int THREADS>
void launch_compute_histograms_template(
    const T* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread
) {
    constexpr int RADIX = RADIX_LOG << 1;
    constexpr int HISTOGRAMS = NBITS / RADIX_LOG;
    
    int blocks = divup(N, THREADS * nreps_per_thread);
    if (!hist_init) {
        auto kernel = compute_histograms<T, HISTOGRAMS, RADIX_LOG, THREADS, false>;
        // if (kernel_blocks.count(&kernel) == 0) {
        //     int numBlocksPerSm = 0;
        //     cudaDeviceProp deviceProp;
        //     cudaGetDeviceProperties(&deviceProp, 0);
        //     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, THREADS, 0);
        //     blocks = min(numBlocksPerSm * deviceProp.multiProcessorCount, blocks);
        //     kernel_blocks[&kernel] = numBlocksPerSm * deviceProp.multiProcessorCount;
        // } else {
        //     blocks = min(kernel_blocks[&kernel], blocks);
        // }

        // cudaMemsetAsync((void *) hist, 0, HISTOGRAMS * RADIX * 4);
        // cudaDeviceSynchronize();
        kernel<<<blocks, THREADS>>>(data, hist, N);
    } else {
        auto kernel = compute_histograms<T, HISTOGRAMS, RADIX_LOG, THREADS, true>;

        if (kernel_blocks.count(&kernel) == 0) {
            int numBlocksPerSm = 0;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, THREADS, 0);
            blocks = min(numBlocksPerSm * deviceProp.multiProcessorCount, blocks);
            kernel_blocks[&kernel] = numBlocksPerSm * deviceProp.multiProcessorCount;
        } else {
            blocks = min(kernel_blocks[&kernel], blocks);
        }
        void *kernelArgs[] = {
            &data, &hist, (void *) &N
        };
        cudaLaunchCooperativeKernel(
            kernel, dim3(blocks, 1, 1), dim3(THREADS, 1, 1), kernelArgs
        );
    }
}

template <typename T, int BITS, int RADIX_LOG>
void launch_compute_histograms_T(
    const T* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int threads
) {
    switch (threads) {
        case 32: 
            launch_compute_histograms_template<T, BITS, RADIX_LOG, 32>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 64: 
            launch_compute_histograms_template<T, BITS, RADIX_LOG, 64>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 128: 
            launch_compute_histograms_template<T, BITS, RADIX_LOG, 128>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 256: 
            launch_compute_histograms_template<T, BITS, RADIX_LOG, 256>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 512: 
            launch_compute_histograms_template<T, BITS, RADIX_LOG, 512>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        default:
            break;
    }
}

template <typename T, int BITS>
void launch_compute_histograms_T_RADIX(
    const T* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int threads, const int radix_log)
{
    switch (radix_log) {
        case 5:
            launch_compute_histograms_T<T, BITS, 5>(data, hist, N, hist_init, nreps_per_thread, threads);
            break;
        case 6:
            launch_compute_histograms_T<T, BITS, 6>(data, hist, N, hist_init, nreps_per_thread, threads);
            break;
        case 7:
            launch_compute_histograms_T<T, BITS, 7>(data, hist, N, hist_init, nreps_per_thread, threads);
            break;
        case 8:
            launch_compute_histograms_T<T, BITS, 8>(data, hist, N, hist_init, nreps_per_thread, threads);
            break;
        case 10:
            launch_compute_histograms_T<T, BITS, 10>(data, hist, N, hist_init, nreps_per_thread, threads);
            break;
        default:
            break;
    }

}

void launch_compute_histograms_b32(
    const uint32_t* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int threads, const int radix_log
) {
    launch_compute_histograms_T_RADIX<uint32_t, 32>(
        data, hist, N, hist_init, nreps_per_thread, threads, radix_log
    );
}


template <typename T, int HISTOGRAMS, int RADIX_LOG>
__device__ __forceinline__ void compute_hists(
    T dreg,
    bool mask,
    uint32_t* s_hists // [HISTOGRAMS * RADIX]
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    for (int h = 0; h < HISTOGRAMS; ++h) {
        uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
        WLMS d_wlms = wlmsMasked<RADIX_LOG>(dh, mask);
        auto cnt = __popc(d_wlms.warp_flags);
        if (d_wlms.bits == 0 && cnt > 0) {
            atomicAdd(s_hists + h * RADIX + dh, cnt);
            // s_hists[h * RADIX + dh] += cnt;
        }
        // __syncwarp();
    }
}

template <typename T, int HISTOGRAMS, int RADIX_LOG, int THREADS, int NPIPE, bool INIT_GLOBAL>
__global__ void __launch_bounds__(THREADS) compute_histograms_pipelined(
    const T *__restrict__ data, // [N]
    uint32_t *__restrict__ g_hists, // [HISTOGRAMS, RADIX]
    const int N
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    static_assert(RADIX_LOG <= 16 && RADIX_LOG >= 2, "");

    __shared__ T s_data[NPIPE][THREADS];
    __shared__ uint32_t s_hists[HISTOGRAMS * RADIX];
    cuda::pipeline<cuda::thread_scope::thread_scope_thread> pipeline = cuda::make_pipeline();

    if (INIT_GLOBAL) {
        for (int i = threadIdx.x + blockIdx.x * THREADS; i < HISTOGRAMS * RADIX; i += THREADS * gridDim.x) {
            g_hists[i] = 0;
        }
    }
    for (int i = threadIdx.x; i < HISTOGRAMS * RADIX; i += THREADS) {
        s_hists[i] = 0;
    }
    for (int i = 0; i < NPIPE; ++i) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &s_data[i][threadIdx.x], &data[offset], (offset < N) ? sizeof(T) : 0, pipeline
        );
        pipeline.producer_commit();
    } 
    __syncthreads();
    const int iters = divup(N, THREADS * gridDim.x);
    for (int i = NPIPE; i < iters; i++) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        const int offset_prev = threadIdx.x + blockIdx.x * THREADS + (i - NPIPE) * THREADS * gridDim.x;
        cuda::pipeline_consumer_wait_prior<NPIPE - 1>(pipeline);

        T dreg = s_data[(i - NPIPE) % NPIPE][threadIdx.x];
        compute_hists<T, HISTOGRAMS, RADIX_LOG>(dreg, offset_prev < N, s_hists);
        pipeline.consumer_release();
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &s_data[(i - NPIPE) % NPIPE][threadIdx.x], &data[offset], (offset < N) ? sizeof(T) : 0, pipeline
        );
        pipeline.producer_commit();
    }
    cuda::pipeline_consumer_wait_prior<0>(pipeline);
    const int it_offset = iters > NPIPE ? iters - NPIPE : 0;
    for (int i = it_offset; i < NPIPE + it_offset; i++) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        T dreg = s_data[i % NPIPE][threadIdx.x];
        compute_hists<T, HISTOGRAMS, RADIX_LOG>(dreg, offset < N, s_hists);
    }

    pipeline.consumer_release();

    __syncthreads();
    if (INIT_GLOBAL) {
        auto grid = cooperative_groups::this_grid();
        grid.sync();
    }

    // use a warp to do scan for each histogram for now
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;    
    for (int i = warp_id; i < HISTOGRAMS; i += THREADS / 32) {
        uint32_t s_accum = 0;
        uint32_t s_first = s_hists[i * RADIX];
        
        for (int j = lane_id; j < RADIX; j += 32) {
            const uint32_t s = InclusiveWarpScan(s_hists[i * RADIX + j]);
            atomicAdd(g_hists + i * RADIX + j, s + s_accum - s_first);
            s_accum += __shfl_sync(0xffffffff, s, 31);
            // atomicAdd(g_hists + i * RADIX + j, s_hists[i * RADIX + j]);
        }
    }
}


template <typename T, int HISTOGRAMS, int RADIX_LOG, int NPIPE>
void launch_compute_histograms_pipelined_template(
    const T* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread
) {
    constexpr int THREADS = 256;
    int blocks = divup(N, THREADS * nreps_per_thread);
    if (!hist_init) {
        auto kernel = compute_histograms_pipelined<T, HISTOGRAMS, RADIX_LOG, THREADS, NPIPE, false>;
        kernel<<<blocks, THREADS>>>(data, hist, N);
    } else {
        auto kernel = compute_histograms_pipelined<T, HISTOGRAMS, RADIX_LOG, THREADS, NPIPE, true>;
        if (kernel_blocks.count(&kernel) == 0) {
            int numBlocksPerSm = 0;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, THREADS, 0);
            blocks = min(numBlocksPerSm * deviceProp.multiProcessorCount, blocks);
            kernel_blocks[&kernel] = numBlocksPerSm * deviceProp.multiProcessorCount;
        } else {
            blocks = min(kernel_blocks[&kernel], blocks);
        }
        void *kernelArgs[] = {
            &data, &hist, (void *) &N
        };
        cudaLaunchCooperativeKernel(
            kernel, dim3(blocks, 1, 1), dim3(THREADS, 1, 1), kernelArgs
        );
    }
}


template <typename T, int HISTOGRAMS, int RADIX_LOG>
void launch_compute_histograms_pipelined_T(
    const T* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int npipe
) {
    switch (npipe) {
        case 2: 
            launch_compute_histograms_pipelined_template<T, HISTOGRAMS, RADIX_LOG, 2>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 3: 
            launch_compute_histograms_pipelined_template<T, HISTOGRAMS, RADIX_LOG, 4>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        case 4: 
            launch_compute_histograms_pipelined_template<T, HISTOGRAMS, RADIX_LOG, 4>(data, hist, N, hist_init, nreps_per_thread); 
            break;
        default:
            break;
    }
}


void launch_compute_histograms_pipelined_b32(
    const uint32_t* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int npipe
) {
    launch_compute_histograms_pipelined_T<uint32_t, 4, 8>(
        data, hist, N, hist_init, nreps_per_thread, npipe
    );
}

template <typename T, int HISTOGRAMS, int RADIX_LOG, int RADIX_DUP>
__global__ void compute_histograms_shared(
    const T *__restrict__ data, // [N]
    uint32_t *__restrict__ g_hists, // [HISTOGRAMS, RADIX]
    const int N,
    const int radix_stride
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    const int THREADS = blockDim.x;
    const int THREADS_PER_RADIX = THREADS / RADIX_DUP;

    
    static_assert(RADIX_LOG <= 16 && RADIX_LOG >= 2, "");

    __shared__ uint32_t s_hists[RADIX_DUP][HISTOGRAMS * RADIX];
    for (int j = threadIdx.x; j < HISTOGRAMS * RADIX * RADIX_DUP; j += THREADS) {
        (&s_hists[0][0])[j] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x + blockIdx.x * THREADS; i < N; i += THREADS * gridDim.x) {
        T dreg = (i < N) ? data[i] : 0;
        for (int h = 0; h < HISTOGRAMS; ++h) {
            uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
            atomicAdd(s_hists[threadIdx.x / THREADS_PER_RADIX] + h * RADIX + dh, 1);
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < HISTOGRAMS * RADIX; j += THREADS) {
        uint32_t radix_reduce = 0;
        for (int i = 0; i < RADIX_DUP; ++i) {
            radix_reduce += s_hists[i][j];
        }
        s_hists[0][j] = radix_reduce;
    }
    __syncthreads();

    // use a warp to do scan for each histogram for now
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;    
    for (int i = warp_id; i < HISTOGRAMS; i += THREADS / 32) {
        uint32_t s_accum = 0;
        uint32_t s_first = s_hists[0][i * RADIX];
        
        for (int j = lane_id; j < RADIX; j += 32) {
            const uint32_t s = InclusiveWarpScan(s_hists[0][i * RADIX + j]);
            atomicAdd(g_hists + i * radix_stride + j, s + s_accum - s_first);
            s_accum += __shfl_sync(0xffffffff, s, 31);
            // atomicAdd(g_hists + i * RADIX + j, s_hists[i * RADIX + j]);
        }
    }
}

void launch_compute_histograms_shared_b32(
    const uint32_t* data, uint32_t* hist, const int N, const int radix_stride, const int nreps_per_thread,
    const int threads, const int radix_dup
) {
    constexpr int RADIX_LOG = 8;
    constexpr int HISTOGRAMS = 4;

    const int blocks = divup(N, threads * nreps_per_thread);
    switch (radix_dup) {
        case 1:
            compute_histograms_shared<uint32_t, HISTOGRAMS, RADIX_LOG, 1><<<blocks, threads>>>(data, hist, N, radix_stride);
            break;
        case 2:
            compute_histograms_shared<uint32_t, HISTOGRAMS, RADIX_LOG, 2><<<blocks, threads>>>(data, hist, N, radix_stride);
            break;
        case 3:
            compute_histograms_shared<uint32_t, HISTOGRAMS, RADIX_LOG, 3><<<blocks, threads>>>(data, hist, N, radix_stride);
            break;
        case 4:
            compute_histograms_shared<uint32_t, HISTOGRAMS, RADIX_LOG, 4><<<blocks, threads>>>(data, hist, N, radix_stride);
            break;
    }
}



template <typename T, int HISTOGRAMS, int RADIX_LOG, int RADIX_DUP, int NPIPE>
__global__ void compute_histograms_shared_pipelined(
    const T *__restrict__ data, // [N]
    uint32_t *__restrict__ g_hists, // [HISTOGRAMS, RADIX]
    const int N,
    const int radix_stride
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    const int THREADS = blockDim.x;
    const int THREADS_PER_RADIX = THREADS / RADIX_DUP;

    
    static_assert(RADIX_LOG <= 16 && RADIX_LOG >= 2, "");

    extern __shared__ T s_data[]; // NPIPE * THREADS
    __shared__ uint32_t s_hists[RADIX_DUP][HISTOGRAMS * RADIX];
    cuda::pipeline<cuda::thread_scope::thread_scope_thread> pipeline = cuda::make_pipeline();

    for (int j = threadIdx.x; j < HISTOGRAMS * RADIX * RADIX_DUP; j += THREADS) {
        (&s_hists[0][0])[j] = 0;
    }

    for (int i = 0; i < NPIPE; ++i) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &s_data[i * THREADS + threadIdx.x], &data[offset], (offset < N) ? sizeof(T) : 0, pipeline
        );
        pipeline.producer_commit();
    }

    __syncthreads();

    // for (int i = threadIdx.x + blockIdx.x * THREADS; i < N; i += THREADS * gridDim.x) {
    //     T dreg = (i < N) ? data[i] : 0;
    //     for (int h = 0; h < HISTOGRAMS; ++h) {
    //         uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
    //         atomicAdd(s_hists[threadIdx.x / THREADS_PER_RADIX] + h * RADIX + dh, 1);
    //     }
    // }

    const int iters = divup(N, THREADS * gridDim.x);
    for (int i = NPIPE; i < iters; i++) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        const int offset_prev = threadIdx.x + blockIdx.x * THREADS + (i - NPIPE) * THREADS * gridDim.x;
        cuda::pipeline_consumer_wait_prior<NPIPE - 1>(pipeline);

        T dreg = s_data[((i - NPIPE) % NPIPE) * THREADS + threadIdx.x];

        if (offset_prev < N) {
            for (int h = 0; h < HISTOGRAMS; ++h) {
                uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
                atomicAdd(s_hists[threadIdx.x / THREADS_PER_RADIX] + h * RADIX + dh, 1);
            }
        }

        pipeline.consumer_release();
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &s_data[((i - NPIPE) % NPIPE) * THREADS + threadIdx.x], &data[offset], (offset < N) ? sizeof(T) : 0, pipeline
        );
        pipeline.producer_commit();
    }
    cuda::pipeline_consumer_wait_prior<0>(pipeline);
    const int it_offset = iters > NPIPE ? iters - NPIPE : 0;
    for (int i = it_offset; i < NPIPE + it_offset; i++) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * THREADS * gridDim.x;
        T dreg = s_data[(i % NPIPE) * THREADS + threadIdx.x];
        if (offset < N) {
            for (int h = 0; h < HISTOGRAMS; ++h) {
                uint32_t dh = (dreg >> (RADIX_LOG * h)) & RADIX_MASK;
                atomicAdd(s_hists[threadIdx.x / THREADS_PER_RADIX] + h * RADIX + dh, 1);
            }
        }
    }

    pipeline.consumer_release();
    __syncthreads();
    if (RADIX_DUP > 1) {   
        for (int j = threadIdx.x; j < HISTOGRAMS * RADIX; j += THREADS) {
            uint32_t radix_reduce = 0;
            for (int i = 0; i < RADIX_DUP; ++i) {
                radix_reduce += s_hists[i][j];
            }
            s_hists[0][j] = radix_reduce;
        }
        __syncthreads();
    }
    // use a warp to do scan for each histogram for now
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;    
    for (int i = warp_id; i < HISTOGRAMS; i += THREADS / 32) {
        uint32_t s_accum = 0;
        uint32_t s_first = s_hists[0][i * RADIX];
        
        for (int j = lane_id; j < RADIX; j += 32) {
            const uint32_t s = InclusiveWarpScan(s_hists[0][i * RADIX + j]);
            atomicAdd(g_hists + i * radix_stride + j, s + s_accum - s_first);
            s_accum += __shfl_sync(0xffffffff, s, 31);
            // atomicAdd(g_hists + i * RADIX + j, s_hists[i * RADIX + j]);
        }
    }
}


template <int NPIPE>
void launch_compute_histograms_shared_pipelined_b32_template(
    const uint32_t* data, uint32_t* hist, const int N, const int radix_stride, 
    const int threads,
    const int nreps_per_thread,
    const int radix_dup
) {
    constexpr int RADIX_LOG = 8;
    constexpr int HISTOGRAMS = 4;
    const int dynamic_smem = NPIPE * threads * sizeof(uint32_t);

    const int blocks = divup(N, threads * nreps_per_thread);
    switch (radix_dup) {
        case 1:
            compute_histograms_shared_pipelined<uint32_t, HISTOGRAMS, RADIX_LOG, 1, NPIPE><<<blocks, threads, dynamic_smem>>>(data, hist, N, radix_stride);
            break;
        case 2:
            compute_histograms_shared_pipelined<uint32_t, HISTOGRAMS, RADIX_LOG, 2, NPIPE><<<blocks, threads, dynamic_smem>>>(data, hist, N, radix_stride);
            break;
        case 3:
            compute_histograms_shared_pipelined<uint32_t, HISTOGRAMS, RADIX_LOG, 3, NPIPE><<<blocks, threads, dynamic_smem>>>(data, hist, N, radix_stride);
            break;
        case 4:
            compute_histograms_shared_pipelined<uint32_t, HISTOGRAMS, RADIX_LOG, 4, NPIPE><<<blocks, threads, dynamic_smem>>>(data, hist, N, radix_stride);
            break;
    }
}


void launch_compute_histograms_shared_pipelined_b32(
    const uint32_t* data, uint32_t* hist, const int N, const int radix_stride, 
    const int threads,
    const int nreps_per_thread,
    const int radix_dup, const int npipe
) {
    switch (npipe) {
        case 2:
            launch_compute_histograms_shared_pipelined_b32_template<2>(data, hist, N, radix_stride, threads, nreps_per_thread, radix_dup);
            break;
        case 3:
            launch_compute_histograms_shared_pipelined_b32_template<3>(data, hist, N, radix_stride, threads, nreps_per_thread, radix_dup);
            break;
        case 4:
            launch_compute_histograms_shared_pipelined_b32_template<4>(data, hist, N, radix_stride, threads, nreps_per_thread, radix_dup);
            break;
        default:
            break;
    }
}


template <typename T, T MaxElem, int RADIX_LOG, int ELEM_PER_THREAD, int WARPS>
__global__ __forceinline__ void radix_pass(
    const T* key, // [N]
    const uint32_t* global_histogram, // [RADIX]
    T* res, // [N]
    uint32_t* hist, // [(NBLOCKS - 1) * RADIX]
    const int N,
    const int bits_offset,
    int* index // [1]
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr T RADIX_MASK = T (RADIX - 1);
    constexpr int THREADS = WARPS * 32;

    constexpr int ELEM_PER_BLOCK = ELEM_PER_THREAD * THREADS;
    constexpr int WARP_HISTS = WARPS * RADIX;
    constexpr int SHARED_DATA = ELEM_PER_BLOCK < WARP_HISTS ? WARP_HISTS * sizeof(uint32_t) : ELEM_PER_BLOCK * sizeof(T);
    __shared__ uint8_t s_data[SHARED_DATA];
    uint32_t* s_hists = (uint32_t*) s_data;
    T* s_key_data = (T*) s_data;

    T key_data[ELEM_PER_THREAD];
    uint16_t offsets[ELEM_PER_THREAD];
    if (threadIdx.x == 0) {
        const int block_id = atomicAdd(index, 1);
        s_hists[0] = block_id;
    }
    __syncthreads();
    const int block_id = s_hists[0];

    auto block = cooperative_groups::this_thread_block();

    { // init shared memory
        for (int i = threadIdx.x; i < WARP_HISTS; i += THREADS) {
            s_hists[i] = 0;
        }
        __syncthreads();
    }
    { // load histograms
        int offset = (threadIdx.x % 32) + (threadIdx.x / 32) * ELEM_PER_THREAD * 32 + block_id * ELEM_PER_BLOCK;
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            if (offset < N) {
                key_data[i] = key[offset];
            } else {
                key_data[i] = MaxElem;
            }
            offset += 32;
        }
    }
    { // wlms
        auto s_warp_hist = threadIdx.x / 32 * RADIX + s_hists;
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            uint8_t bin = key_data[i] >> bits_offset & (uint8_t(RADIX_MASK));
            WLMS wlms_t = wlms<RADIX_LOG>(bin);
            uint32_t inc_val;
            if (wlms_t.bits == 0) {
                // probably don't need to do atomic add here
                inc_val = atomicAdd(s_warp_hist + bin, __popc(wlms_t.warp_flags));
            }
            offsets[i] = __shfl_sync(0xffffffff, inc_val, __ffs(wlms_t.warp_flags) - 1) + wlms_t.bits;
        }
        __syncthreads();
    }

    __shared__ uint32_t s_local_hist[RADIX];
    { // sum up the warp hists
        auto s_warp_hist = threadIdx.x / 32 * RADIX + s_hists;
        static_assert(THREADS >= RADIX, "");
        uint32_t reduction;
        auto gp = cg::tiled_partition<RADIX>(block);
        if (threadIdx.x < RADIX) {
            // exclusive scan, bin-wise across histograms
            reduction = s_warp_hist[threadIdx.x];
            for (int i = 1; i < WARPS; ++i) {
                auto v = s_warp_hist[i * RADIX + threadIdx.x];
                s_warp_hist[i * RADIX + threadIdx.x] = reduction;
                reduction += v;
            }
            
            atomicAdd(hist + (block_id + 1) * RADIX + threadIdx.x, reduction << 2 | 1);
            reduction = cg::exclusive_scan(gp, reduction);
        }
        __syncthreads(); // not necessary to due to synchonization in exclusive_scan
        if (threadIdx.x < RADIX) {
            s_local_hist[threadIdx.x] = reduction; // reuse the first warp hist
        }
        __syncthreads();

        if (threadIdx.x / 32 == 0) { // first warp is zero for exclusive scan
            for (int i = 0; i < ELEM_PER_THREAD; ++i) {
                uint8_t bin = key_data[i] >> bits_offset & (uint8_t(RADIX_MASK));
                offsets[i] += s_local_hist[bin];
            }
        } else {
            for (int i = 0; i < ELEM_PER_THREAD; ++i) {
                uint8_t bin = key_data[i] >> bits_offset & (uint8_t(RADIX_MASK));
                offsets[i] += s_local_hist[bin] + s_warp_hist[bin];
            }
        }
        // now offsets contain the local offsets per block
        __syncthreads();
    }

    { // scatter keys into shared
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            assert(offsets[i] < ELEM_PER_BLOCK);
            s_key_data[offsets[i]] = key_data[i];
            // s_key_data[i * THREADS + threadIdx.x] = offsets[i];
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < ELEM_PER_BLOCK; ++i) {
                printf("%d ", s_key_data[i]);
            }
        }
    }

    if (threadIdx.x < RADIX) { // lookback
        const int nblocks = divup(N, ELEM_PER_BLOCK);
        uint32_t reduction = 0;
        
        for (int k = block_id; k >= 0; ) {
            assert(k * RADIX + threadIdx.x < (nblocks + 1) * RADIX);
            uint32_t flag_key;
            if (k == 0) {
                flag_key = global_histogram[threadIdx.x] << 2 | 2;
            } else {
                flag_key = hist[k * RADIX + threadIdx.x];
            }

            if ((flag_key & 3) == 2) {
                reduction += flag_key >> 2;
                if (k < nblocks - 1) {
                    atomicAdd(hist + threadIdx.x + k * RADIX, reduction << 2 | 1);
                }
                s_local_hist[threadIdx.x] = reduction - s_local_hist[threadIdx.x];
                break;
            }

            if ((flag_key & 3) == 1) {
                reduction += flag_key >> 2;
                k -= 1;
            }
        }
    }
    __syncthreads();

    { // scatter keys into global
        for (int i = threadIdx.x; i < ELEM_PER_BLOCK; i += THREADS) {
            auto key = s_key_data[i];
            const int global_offset = i + s_local_hist[(key >> bits_offset & (uint8_t(RADIX_MASK)))];
            if (global_offset < N) { // to account for padding
                res[global_offset] = key;
            }
        }
    }
    __syncthreads();
}




void launch_single_radix_pass_b32(
    const uint32_t* key, // [N]
    const uint32_t* global_histogram, // [RADIX]
    uint32_t* res, // [N]
    uint32_t* hist, // [(NBLOCKS + 1) * RADIX]
    const int N,
    const int bits_offset,
    int* index // [1]
) {
    constexpr int radix_log = 8;
    constexpr int elem_per_thread = 15;
    constexpr int threads = 256;

    auto kernel = radix_pass<uint32_t, uint32_t(0xFFFFFFFF), radix_log, elem_per_thread, threads / 32>;
    int blocks = divup(N, threads * elem_per_thread);
    kernel<<<blocks, threads>>>(
        key, global_histogram, res, hist, N, bits_offset, index
    );
}



template <typename T, int HISTOGRAMS, int RADIX_LOG, int ELEM_PER_THREAD, int WARPS>
__device__ __forceinline__ void compute_global_local_hist(
    const T* sort,       // [N]
    uint32_t* global_histogram, // [HISTOGRAMS * RADIX]
    uint32_t* block_histogram,   // [HISTOGRAMS * binningThreadBlocks * RADIX]
    const int bits_offsets[HISTOGRAMS],
    const int N
) {
    static_assert(RADIX_LOG <= 8, "radix must fit in 8 bits");
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr T RADIX_MASK = T (RADIX - 1);
    constexpr int THREADS = WARPS * 32;
    static_assert(THREADS >= RADIX, "");
    constexpr int ELEM_PER_BLOCK = ELEM_PER_THREAD * WARPS * 32;

    auto grid = cooperative_groups::this_grid();
    __shared__ uint32_t local_hist[HISTOGRAMS][RADIX];
    for (int i = threadIdx.x; i < RADIX * HISTOGRAMS; i += THREADS) {
        local_hist[i / RADIX][i % RADIX] = 0;
    }
    for (int i = threadIdx.x + blockIdx.x * THREADS; i < RADIX * HISTOGRAMS; i += THREADS * gridDim.x) {
        global_histogram[i] = 0;
    }
    __syncthreads();
    grid.sync();
    const int grid_stride = ELEM_PER_BLOCK * gridDim.x;
    const int binning_thread_blocks = divup(N, ELEM_PER_BLOCK);
    uint32_t global_count[HISTOGRAMS];
    for (int h = 0; h < HISTOGRAMS; ++h) {
        global_count[h] = 0;
    }

    for (int it = 0; it < divup(N, grid_stride); it++) { // launch enough blocks to achieve full concurrency
        // [blockIdx.x, ELEM_PER_THREAD, threadIdx.x]
        const int offset_block = it * grid_stride + blockIdx.x * ELEM_PER_BLOCK;
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            const int offset = offset_block + i * THREADS + threadIdx.x;
            T key = offset < N ? sort[offset] : RADIX_MASK;

            for (int h = 0; h < HISTOGRAMS; ++h) {
                uint8_t bin = key >> bits_offsets[h] & (uint8_t(RADIX_MASK));
                WLMS wlms_t = wlms<RADIX_LOG>(bin);
                if (wlms_t.bits == 0) {
                    atomicAdd(local_hist[h] + bin, __popc(wlms_t.warp_flags));
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < RADIX) {
            for (int h = 0; h < HISTOGRAMS; ++h) {
                global_count[h] += local_hist[h][threadIdx.x];
                int offset_hist = it * gridDim.x * RADIX + blockIdx.x * RADIX;
                block_histogram[offset_hist + h * binning_thread_blocks * RADIX + threadIdx.x] = local_hist[h][threadIdx.x];
                local_hist[h][threadIdx.x] = 0;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < RADIX) {
        for (int h = 0; h < HISTOGRAMS; ++h) {
            atomicAdd(global_histogram + h * RADIX + threadIdx.x, global_count[h]);
        }
    }
    grid.sync();
}

template <typename T, T MAX, int RADIX_LOG, int ELEM_PER_THREAD, int WARPS>
__device__ __forceinline__ void compute_local_offset_(
    const T* sort,       // [N]
    const uint32_t* global_histogram, // [RADIX]
    uint32_t* block_histogram,   // [binningThreadBlocks * RADIX]
    T* sort_payload,     // [N]
    const int bits_offset,
    const int N
) {
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr T RADIX_MASK = T (RADIX - 1);
    constexpr int THREADS = WARPS * 32;
    static_assert(THREADS >= RADIX, "");
    constexpr int ELEM_PER_BLOCK = ELEM_PER_THREAD * WARPS * 32;
    const int grid_stride = ELEM_PER_BLOCK * gridDim.x;
    const int binning_thread_blocks = divup(N, ELEM_PER_BLOCK);
    const int block_reps = divup(N, grid_stride);

    // sort is partitioned like this: 
    // [gridDim.x, block_reps, ELEM_PER_THREAD, blockDim.x]
    // [blockIdx.x, 1, 1, threadIdx.x]
    for (int it = 0; it < block_reps; ++it) {
        uint16_t offsets[ELEM_PER_THREAD];
        T keys[ELEM_PER_THREAD];

        const int offset_block = blockIdx.x * block_reps * ELEM_PER_BLOCK + it * ELEM_PER_BLOCK;
        for (int i = 0; i < ELEM_PER_THREAD; ++i) {
            const int offset = offset_block + i * THREADS + threadIdx.x;
            keys[i] = offset < N ? sort[offset] : MAX;
        }
        
    }

}
