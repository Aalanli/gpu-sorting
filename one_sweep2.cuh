#include <stdint.h>
#include "Utils.cuh"
#include <cuda/pipeline>


constexpr int FLAG_REDUCTION = 1;
constexpr int FLAG_INCLUSIVE = 2;
constexpr int FLAG_MASK = 3;
// constexpr int LANE_LOG = 5;

template <typename T, int RADIX_LOG, int ELEM_PER_THREAD, int THREADS>
__global__ void DigitBinningPassKeysOnly(
    const T* sort,
    T* alt,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift)
{
    constexpr int RADIX = 1 << RADIX_LOG;
    constexpr int RADIX_MASK = RADIX - 1;
    constexpr int KEY_DATA_SIZE = ELEM_PER_THREAD * THREADS;
    constexpr int WARP_HIST_SIZE = THREADS / 32 * RADIX;
    constexpr int SHARED_BYTES = KEY_DATA_SIZE * sizeof(T) < WARP_HIST_SIZE * sizeof(uint32_t) ? WARP_HIST_SIZE * sizeof(uint32_t) : KEY_DATA_SIZE * sizeof(T);

    __align__(4) __shared__ uint8_t s_data[SHARED_BYTES];

    uint32_t* s_warpHistograms = (uint32_t*)s_data;
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < WARP_HIST_SIZE; i += blockDim.x)  //unnecessary work for last partion but still a win to avoid another barrier
        s_warpHistograms[i] = 0;

    //atomically assign partition tiles
    if (threadIdx.x == 0)
        s_localHistogram[0] = atomicAdd((uint32_t*)index, 1);
    __syncthreads();
    const uint32_t partitionIndex = s_localHistogram[0];

    //load keys
    T keys[ELEM_PER_THREAD];
    if (partitionIndex < gridDim.x - 1)
    {
        for (uint32_t i = 0, t = getLaneId() + (WARP_INDEX * 32 * ELEM_PER_THREAD) + partitionIndex * THREADS * ELEM_PER_THREAD; i < ELEM_PER_THREAD; ++i, t += 32)
            keys[i] = sort[t];
    }

    //To handle input sizes not perfect multiples of the partition tile size,
    //load "dummy" keys, which are keys with the highest possible digit.
    //Because of the stability of the sort, these keys are guaranteed to be 
    //last when scattered. This allows for effortless divergence free sorting
    //of the final partition.
    if (partitionIndex == gridDim.x - 1)
    {
        for (uint32_t i = 0, t = getLaneId() + (WARP_INDEX * 32 * ELEM_PER_THREAD) + partitionIndex * THREADS * ELEM_PER_THREAD; i < ELEM_PER_THREAD; ++i, t += 32)
            keys[i] = t < size ? sort[t] : 0xffffffff;
    }

    //WLMS
    uint16_t offsets[ELEM_PER_THREAD];
    for (uint32_t i = 0; i < ELEM_PER_THREAD; ++i)
    {
        unsigned warpFlags = 0xffffffff;
        for (int k = 0; k < RADIX_LOG; ++k)
        {
            const bool t2 = keys[i] >> k + radixShift & 1;
            warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const uint32_t bits = __popc(warpFlags & getLaneMaskLt());

        uint32_t preIncrementVal;
        if (bits == 0)
            preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));

        offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
    }
    __syncthreads();

    //exclusive prefix sum up the warp histograms
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = s_warpHistograms[threadIdx.x];
        for (uint32_t i = threadIdx.x + RADIX; i < WARP_HIST_SIZE; i += RADIX)
        {
            reduction += s_warpHistograms[i];
            s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }

        atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX],
            FLAG_REDUCTION | reduction << 2);

        //begin the exclusive prefix sum across the reductions
        s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    }
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
        s_localHistogram[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
    __syncthreads();

    if (threadIdx.x < RADIX && getLaneId())
        s_localHistogram[threadIdx.x] += __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
    __syncthreads();

    //update offsets
    if (WARP_INDEX)
    {
        for (uint32_t i = 0; i < ELEM_PER_THREAD; ++i)
        {
            const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
        }
    }
    else
    {
        for (uint32_t i = 0; i < ELEM_PER_THREAD; ++i)
            offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
    }
    __syncthreads();

    //scatter keys into shared memory

    T* s_key_data = (T*) s_data;

    for (uint32_t i = 0; i < ELEM_PER_THREAD; ++i)
        s_key_data[offsets[i]] = keys[i];

    //lookback
    static_assert(THREADS >= RADIX, "THREADS must be greater than or equal to RADIX");
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = 0;
        for (uint32_t k = partitionIndex; k >= 0; )
        {
            const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];

            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE)
            {
                reduction += flagPayload >> 2;
                atomicAdd((uint32_t*)&passHistogram[threadIdx.x + (partitionIndex + 1) * RADIX], 1 | (reduction << 2));
                s_localHistogram[threadIdx.x] = reduction - s_localHistogram[threadIdx.x];
                break;
            }

            if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION)
            {
                reduction += flagPayload >> 2;
                k--;
            }
        }
    }
    __syncthreads();

    //scatter runs of keys into device memory
    if (partitionIndex < gridDim.x - 1)
    {
        for (uint32_t i = threadIdx.x; i < KEY_DATA_SIZE; i += blockDim.x)
            alt[s_localHistogram[s_key_data[i] >> radixShift & RADIX_MASK] + i] = s_key_data[i];
    }

    if (partitionIndex == gridDim.x - 1)
    {
        const uint32_t finalPartSize = size - partitionIndex * THREADS * ELEM_PER_THREAD;
        for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x)
            alt[s_localHistogram[s_key_data[i] >> radixShift & RADIX_MASK] + i] = s_key_data[i];
    }
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
        if (blockIdx.x == 0 && lane_id == 0) {
            g_hists[i * radix_stride] = 2;
        }        
        for (int j = lane_id; j < RADIX; j += 32) {
            const uint32_t s = InclusiveWarpScan(s_hists[0][i * RADIX + j]);
            auto accum = (s + s_accum) << 2;
            if (blockIdx.x == 0) {
                accum |= 2;
            }
            if (j < RADIX - 1) {
                atomicAdd(g_hists + i * radix_stride + j + 1, accum);
            }
            s_accum += __shfl_sync(0xffffffff, s, 31);
            // atomicAdd(g_hists + i * RADIX + j, s_hists[i * RADIX + j]);
        }
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
            auto accum = (s + s_accum - s_first) << 2;
            if (blockIdx.x == 0) {
                accum |= FLAG_INCLUSIVE;
            }
            atomicAdd(g_hists + i * radix_stride + j, accum);
            s_accum += __shfl_sync(0xffffffff, s, 31);
            // atomicAdd(g_hists + i * RADIX + j, s_hists[i * RADIX + j]);
        }
    }
}


template <typename T, int HISTOGRAMS, int RADIX_LOG>
void launch_compute_histograms(
    const T* data, // [N]
    uint32_t* g_hists, // [HISTOGRAMS, RADIX...]
    const int N,
    const int radix_stride,
    CUstream_st* stream
) {
    const int reps_per_thread = 24;
    const int threads = 512;
    const int blocks = divup(N, threads * reps_per_thread);
    if (N < 2500000) {
        compute_histograms_shared<T, HISTOGRAMS, RADIX_LOG, 1><<<blocks, threads, 0, stream>>>(
            data, g_hists, N, radix_stride
        );
    } else {
        const int smem_dynamic = threads * sizeof(T) * 3;
        compute_histograms_shared_pipelined<T, HISTOGRAMS, RADIX_LOG, 1, 3><<<blocks, threads, smem_dynamic, stream>>>(
            data, g_hists, N, radix_stride
        );
    }
}

// void launch_compute_histograms_b32(
//     const T* data, // [N]
//     uint32_t* g_hists, // [HISTOGRAMS, RADIX...]
//     const int N,
//     const int radix_stride,
//     CUstream_st* stream
// ) {
//     constexpr int HISTOGRAMS = 4;
//     constexpr int RADIX_LOG = 8;

//     const int reps_per_thread = 24;
//     const int threads = 512;
//     const int blocks = divup(N, threads * reps_per_thread);
//     if (N < 2500000) {
//         compute_histograms_shared<uint32_t, HISTOGRAMS, RADIX_LOG, 1><<<blocks, threads, 0, stream>>>(
//             data, g_hists, N, radix_stride
//         );
//     } else {
//         const int smem_dynamic = threads * sizeof(uint32_t) * 3;
//         compute_histograms_shared_pipelined<uint32_t, HISTOGRAMS, RADIX_LOG, 1, 3><<<blocks, threads, smem_dynamic, stream>>>(
//             data, g_hists, N, radix_stride
//         );
//     }
// }

template <typename T>
void print_device_mem(const T* data, int N) {
    T* h_data = new T[N];
    cudaMemcpy(h_data, data, N * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    delete[] h_data;
}

template <typename T>
class AbstractLauncher {
public:
    virtual int get_workspace_size_in_bytes(int N) = 0;
    virtual void launch(
        const T* data, // [N]
        T* out, // empty[N]
        uint8_t* workspace, // empty[workspace_size]
        const int N,
        CUstream_st* stream
    ) = 0;
};

template <typename T, int SORTING_BITS, int RADIX_LOG, int THREADS, int ELEM_PER_THREAD>
class OneSweepLauncher: public AbstractLauncher<T> {
    static int get_blocks(int N) {
        return divup(N, THREADS * ELEM_PER_THREAD);
    }

    static int get_radix_stride(int blocks) {
        return (1 << RADIX_LOG) * (blocks + 1);
    }

    static int radix() {
        return 1 << RADIX_LOG;
    }

    static constexpr int radix_iters() {
        return (SORTING_BITS + RADIX_LOG - 1) / RADIX_LOG;
    }

public:
    int get_workspace_size_in_bytes(int N) {
        const int blocks = get_blocks(N);
        const int radix_stride = get_radix_stride(blocks);
        const int g_hist_size = radix_stride * radix_iters() * sizeof(uint32_t);

        const int index_size = radix_iters() * sizeof(uint32_t);
        const int buf_size = N * sizeof(T);
        
        return g_hist_size + index_size + buf_size;
    }

    void launch(
        const T* data, // [N]
        T* out, // empty[N]
        uint8_t* workspace, // empty[workspace_size]
        const int N,
        CUstream_st* stream
    ) override {
        const int blocks = get_blocks(N);
        const int radix_stride = get_radix_stride(blocks);

        const int g_hist_size = radix_stride * radix_iters() * sizeof(uint32_t);
        const int index_size = radix_iters() * sizeof(uint32_t);
        const int buf_size = N * sizeof(T);

        uint32_t* g_hists = reinterpret_cast<uint32_t*>(workspace);
        uint32_t* index = reinterpret_cast<uint32_t*>(workspace + g_hist_size);
        T* buf = reinterpret_cast<T*>(workspace + g_hist_size + index_size);
        
        cudaMemsetAsync((void*) workspace, 0, g_hist_size + index_size, stream);
        cudaStreamSynchronize(stream);

        launch_compute_histograms<T, radix_iters(), RADIX_LOG>(
            data, g_hists, N, radix_stride, stream
        );
        
        auto kernel = DigitBinningPassKeysOnly<T, RADIX_LOG, ELEM_PER_THREAD, THREADS>;
        T* src;
        T* dst;

        static_assert(radix_iters() > 0, "radix_iters() must be greater than 0");
        if (radix_iters() % 2 == 0) {
            dst = buf;
        } else {
            dst = out;
        }

        for (int i = 0; i < radix_iters(); ++i) {
            const int radix_shift = i * RADIX_LOG;
            if (i == 0) {
                src = (T*) data;
            }

            kernel<<<blocks, THREADS, 0, stream>>>(
                (const T*) src, dst, (volatile uint32_t*)g_hists + radix_stride * i, (volatile uint32_t*)index + i, N, radix_shift
            );

            if (i == 0) {
                if (radix_iters() % 2 == 0) {
                    src = out;
                } else {
                    src = buf;
                }
            }

            std::swap(src, dst);
        }
    }
};


