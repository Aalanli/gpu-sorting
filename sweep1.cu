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
        s_localHistogram[WARP_HIST_SIZE - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
    __syncthreads();
    const uint32_t partitionIndex = s_warpHistograms[WARP_HIST_SIZE - 1];

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