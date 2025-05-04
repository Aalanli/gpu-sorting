
__global__ void OneSweep::DigitBinningPassKeysOnly(
    uint32_t* sort,
    uint32_t* alt,
    volatile uint32_t* passHistogram,
    volatile uint32_t* index,
    uint32_t size,
    uint32_t radixShift)
{
    __shared__ uint32_t s_warpHistograms[BIN_PART_SIZE];
    __shared__ uint32_t s_localHistogram[RADIX];
    volatile uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << RADIX_LOG];

    //clear shared memory
    for (uint32_t i = threadIdx.x; i < BIN_HISTS_SIZE; i += blockDim.x)  //unnecessary work for last partion but still a win to avoid another barrier
        s_warpHistograms[i] = 0;

    //atomically assign partition tiles
    if (threadIdx.x == 0) {
        s_warpHistograms[BIN_PART_SIZE - 1] = atomicAdd((uint32_t*)&index[radixShift >> 3], 1);
    }
    __syncthreads();
    const uint32_t partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

    //load keys
    uint32_t keys[BIN_KEYS_PER_THREAD];
    if (partitionIndex < gridDim.x - 1)
    {
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = sort[t];
    }

    //To handle input sizes not perfect multiples of the partition tile size,
    //load "dummy" keys, which are keys with the highest possible digit.
    //Because of the stability of the sort, these keys are guaranteed to be 
    //last when scattered. This allows for effortless divergence free sorting
    //of the final partition.
    if (partitionIndex == gridDim.x - 1)
    {
        for (uint32_t i = 0, t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START; i < BIN_KEYS_PER_THREAD; ++i, t += LANE_COUNT)
            keys[i] = t < size ? sort[t] : 0xffffffff;
    }

    //WLMS
    uint16_t offsets[BIN_KEYS_PER_THREAD];
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
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
        for (uint32_t i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE; i += RADIX)
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
    // if (threadIdx.x == 0) {
    //     printf("s_localHistogram: \n");
    //     for (int i = 0; i < RADIX; i++) {
    //         printf("%d ", s_localHistogram[i]);
    //     }
    // }

    //update offsets
    if (WARP_INDEX)
    {
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        {
            const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
            offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
        }
    }
    else
    {
        for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
            offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
    }
    __syncthreads();

    //scatter keys into shared memory
    for (uint32_t i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        s_warpHistograms[offsets[i]] = keys[i];

    //lookback
    if (threadIdx.x < RADIX)
    {
        uint32_t reduction = 0;
        for (int k = partitionIndex; k >= 0; )
        {
            const uint32_t flagPayload = passHistogram[threadIdx.x + k * RADIX];
            // if (threadIdx.x == 1) {
            //     printf("flagPayload: %u, %d\n", flagPayload, k);
            // }
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
    // if (threadIdx.x == 0) {
    //     for (int i = 0; i < 512; i++) {
    //         printf("(%d, %d)", s_warpHistograms[i], s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK]);
    //     }
    //     printf("\n");
    // }

    //scatter runs of keys into device memory
    if (partitionIndex < gridDim.x - 1)
    {
        for (uint32_t i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
            alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    }

    if (partitionIndex == gridDim.x - 1)
    {
        const uint32_t finalPartSize = size - BIN_PART_START;
        for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x)
            alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
    }
}