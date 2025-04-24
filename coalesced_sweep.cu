#include <cooperative_groups.h>

static constexpr int FLAG_NOT_READY = 0;
static constexpr int FLAG_REDUCTION = 1;
static constexpr int FLAG_INCLUSIVE = 2;
static constexpr int FLAG_MASK = 3;


__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
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

template <int NBits>
__device__ __forceinline__ WLMS wlms(uint8_t key) {
    unsigned warp_flags = 0xffffffff;

    for (int k = 0; k < NBits; ++k) {
        const bool t2 = key >> k & 1;
        warp_flags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
    }
    const uint32_t bits = __popc(warp_flags & getLaneMaskLt());
    return WLMS { warp_flags, bits };
}

__device__ __forceinline__ int divup(int a, int b) {
    return (a + b - 1) / b;
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
