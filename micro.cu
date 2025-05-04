#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>

namespace cg = cooperative_groups;

#define THREADS 256

__device__ __forceinline__ int divup(int a, int b) {
    return (a + b - 1) / b;
}

// __device__ __forceinline__ float compute(float x, half2 y, int z, int Iters) {
//     for (int i = 0; i < Iters; ++i) {
//         x += __half22float2(y).x * x + float(z);
//         y.x += (y.x + y.y) / (y.x * y.x + y.y * y.y);
//         y.y += (y.x + y.y) / (y.x * y.x + y.y * y.y);
//         z = z ^ ((int*) &y)[0] & ((int*) &x)[0] || z;
//     }
//     return x + __half22float2(y).x - __half22float2(y).y + float(z);
// }

// __device__ __forceinline__ float compute(float d, int Iters) {
//     return compute(d, __float22half2_rn({d + 1.0f, d + 2.0f}), int(d), Iters);
// }

template <int Iters>
__device__ __forceinline__ float compute(float x) {
    float res = 0.0f;
    // #pragma unroll
    for (int i = 0; i < Iters; ++i) {
        res += x * x + x;
        x += (x + res) / (x * x + res * res);
    }
    return res;
}

template <int Iters>
__global__ void baseline(float* __restrict__ in, float* __restrict__ out, int N) {
    float res = 0.0f;
    for (int i = threadIdx.x + blockIdx.x * THREADS; i < N; i += gridDim.x * THREADS) {
        float d = in[i];
        res += compute<Iters>(d);
    }

    const int offset = threadIdx.x + blockIdx.x * THREADS;  
    if (offset < N) {
        out[offset] = res;
    }
}

template <int Iters>
void launch_baseline(float* in, float* out, int N, int reps) {
    const int blocks = (N + THREADS * reps - 1) / (THREADS * reps);
    baseline<Iters><<<blocks, THREADS>>>(in, out, N);
}


template <int NPipe, int Iters> 
__global__ void reg_pipeline(float* __restrict__ in, float* __restrict__ out, int N) {
    float d[NPipe];
    for (int i = 0; i < NPipe; ++i) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * gridDim.x * THREADS;
        if (offset < N) {
            d[i] = in[offset];
        }
    }
    float res = 0.0f;
    for (int i = NPipe; i < divup(N, gridDim.x * THREADS) + NPipe; i += 1) {
        const int offsetC = threadIdx.x + blockIdx.x * THREADS + (i - NPipe) * gridDim.x * THREADS;
        const int offsetL = threadIdx.x + blockIdx.x * THREADS + i * gridDim.x * THREADS;
        if (offsetC < N) {
            res += compute<Iters>(d[(i - NPipe) % NPipe]);
        }
        if (offsetL < N) {
            d[i % NPipe] = in[offsetL];
        }
    }
    const int offset = threadIdx.x + blockIdx.x * THREADS;
    if (offset < N) {
        out[offset] = res;
    }
}


template <int Iters>    
void launch_reg_pipeline(float* in, float* out, int N, int reps) {
    const int blocks = (N + THREADS * reps - 1) / (THREADS * reps);
    reg_pipeline<2, Iters><<<blocks, THREADS>>>(in, out, N);
    reg_pipeline<3, Iters><<<blocks, THREADS>>>(in, out, N);
    reg_pipeline<4, Iters><<<blocks, THREADS>>>(in, out, N);
}

template <int NPipe, int Iters>
__global__ void shared_pipeline(float* __restrict__ in, float* __restrict__ out, int N) {
    __align__(16) __shared__ float d[NPipe][THREADS];
    // auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope::thread_scope_thread> pipeline = cuda::make_pipeline();
    for (int i = 0; i < NPipe; ++i) {
        const int offset = threadIdx.x + blockIdx.x * THREADS + i * gridDim.x * THREADS;
        pipeline.producer_acquire();
        // cg::memcpy_async(block, &d[i][threadIdx.x], in + offset, (offset < N) ? sizeof(float) : 0);
        cuda::memcpy_async(&d[i][threadIdx.x], in + offset, (offset < N) ? sizeof(float) : 0, pipeline);
        pipeline.producer_commit();
    }

    float res = 0.0f;
    for (int i = NPipe; i < divup(N, gridDim.x * THREADS) + NPipe; i += 1) {
        const int offsetC = threadIdx.x + blockIdx.x * THREADS + (i - NPipe) * gridDim.x * THREADS;
        const int offsetL = threadIdx.x + blockIdx.x * THREADS + i * gridDim.x * THREADS;
        
        cuda::pipeline_consumer_wait_prior<NPipe - 1>(pipeline);

        if (offsetC < N) {
            res += compute<Iters>(d[(i - NPipe) % NPipe][threadIdx.x]);
        }
        pipeline.consumer_release();


        pipeline.producer_acquire();
        cuda::memcpy_async(&d[i % NPipe][threadIdx.x], in + offsetL, (offsetL < N) ? sizeof(float) : 0, pipeline); 
        pipeline.producer_commit(); 
        
        

    }
    const int offset = threadIdx.x + blockIdx.x * THREADS;
    if (offset < N) {
        out[offset] = res;
    }
}


template <int Iters>
void launch_shared_pipeline(float* in, float* out, int N, int reps) {
    const int blocks = (N + THREADS * reps - 1) / (THREADS * reps);
    shared_pipeline<2, Iters><<<blocks, THREADS>>>(in, out, N);
    shared_pipeline<3, Iters><<<blocks, THREADS>>>(in, out, N);
    shared_pipeline<4, Iters><<<blocks, THREADS>>>(in, out, N);
}


template <int Iters> 
__global__ void warp_specialize_reg(float* __restrict__ in, float* __restrict__ out, int N) {
    __align__(16) __shared__ float d[THREADS];
    const int warpId = threadIdx.x / 32;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    if (warpId % 2 == 0){
        const int offset = warpId * 32 + blockIdx.x * THREADS / 2;
        // if (offset < N) {
        //     d[threadIdx.x] = in[offset];
        // }
        cg::memcpy_async(warp, d + warpId * 32, in + offset, 32);
    }
    float res = 0.0f;
    for (int i = 0; i < divup(N, gridDim.x * THREADS / 2); i += 1) {
        const int offsetL = threadIdx.x + blockIdx.x * THREADS / 2 + ((i + 1) / 2) * gridDim.x * THREADS / 2;
        
        if ((warpId + i) % 2 == 0) {
            cg::wait_prior<1>(warp);
            res += compute<Iters>(d[threadIdx.x]);
        } else {
            // d[threadIdx.x] = in[offsetL];
            cg::memcpy_async(warp, d + warpId * 32, in + offsetL, 32);
        }
    }
    const int offset = threadIdx.x + blockIdx.x * THREADS / 2;
    if (offset < N) {
        out[offset] = res;
    }
}

template <int Iters>
void launch_warp_specialize_reg(float* in, float* out, int N, int reps) {
    const int blocks = ( N + THREADS * reps - 1) / (THREADS * reps);
    warp_specialize_reg<Iters><<<blocks, THREADS>>>(in, out, N);
}   


__device__ __forceinline__ void mbarrier_init(int64_t* mbarrier, uint32_t count) {
    asm volatile ("mbarrier.init.shared.b64 [%0], %1;" : : "r"(
        static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier))
    ), "r"(count) : "memory");
}

__device__ __forceinline__ void cp_mbarrier_arrive(int64_t* mbarrier) {
    asm volatile ("cp.async.mbarrier.arrive.shared.b64 [%0];" : : "r"(
        static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier))) : "memory");
}

__device__ __forceinline__ uint64_t mbarrier_arrive(int64_t* mbarrier) {
    uint64_t token = 0;
    asm volatile ("mbarrier.arrive.shared.b64 %0, [%1];" :: "l"(token), "r"(
        static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier))) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int64_t* mbarrier, uint64_t token) {
    asm volatile ("{\n\t"
                  ".reg .pred p;\n\t"
                  "waitLoop: \n\t"
                  "mbarrier.test_wait.shared.b64 p, [%0], %1;\n\t"
                  "@!p nanosleep.u32 50;\n\t"
                  "@!p bra waitLoop;\n\t"
                  "}\n\t"
                  : : 
                  "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier))), "l"(token) : "memory");
}

__device__ __forceinline__ void cp_async_4(uint32_t* dst, uint32_t* src) {
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 4, 4;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(static_cast<uint64_t>(__cvta_generic_to_global(src))) : "memory");
}

template <int NPipe, int Iters> 
__global__ void warp_specialize_shared(float* __restrict__ in, float* __restrict__ out, int N) {
    __align__(16) __shared__ float d[NPipe][THREADS / 2];
    const int warpId = threadIdx.x / 32;
    const int barId = warpId / 2;
    const bool producer = (warpId % 2 == 0);
    const bool consumer = (warpId % 2 == 1);

    const int laneId = threadIdx.x % 32;
    // auto block = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(block);
    // using barrier = cuda::barrier<cuda::thread_scope_block>;
    // __shared__ barrier bar[NPipe][THREADS / 32 / 2];
    __shared__ int64_t bar[NPipe][THREADS / 32 / 2];
    if (laneId == 0 && producer) {
        for (int i = 0; i < NPipe; ++i) {
            // bar[i][barId].init(32);
            mbarrier_init(&bar[i][barId], 32);
        }
    }
    __syncthreads();

    if (producer){
        for (int p = 0; p < NPipe - 1; ++p) {
            const int offset = barId * 32 + laneId + blockIdx.x * THREADS / 2 + p * gridDim.x * THREADS / 2;
            // cuda::memcpy_async(&d[p][barId * 32 + laneId], in + offset, (offset < N) ? sizeof(float) : 0);
            // bar[p][barId].arrive();
            if (offset < N) {
                cp_async_4(reinterpret_cast<uint32_t*>(&d[p][barId * 32 + laneId]), reinterpret_cast<uint32_t*>(in + offset));
                cp_mbarrier_arrive(&bar[p][barId]);
            }
        }
    }
    float res = 0.0f;
    const int iters = divup(N, gridDim.x * THREADS / 2);
    for (int i = NPipe - 1; i < iters + NPipe - 1; i += 1) {
        
        if (consumer) {
            // bar[(i - NPipe + 1) % NPipe][barId].wait();
            uint64_t token = mbarrier_arrive(&bar[(i - NPipe + 1) % NPipe][barId]);
            mbarrier_wait(&bar[(i - NPipe + 1) % NPipe][barId], token);
            res += compute<Iters>(d[(i - NPipe + 1) % NPipe][barId * 32 + laneId]);
        } else if (i < iters) {
            const int offsetL = barId * 32 + laneId + blockIdx.x * THREADS / 2 + i * gridDim.x * THREADS / 2;
            // cuda::memcpy_async(&d[i % NPipe][barId * 32 + laneId], in + offsetL, (offsetL < N) ? sizeof(float) : 0);
            // bar[i % NPipe][barId].arrive();
            if (offsetL < N) {
                cp_async_4(reinterpret_cast<uint32_t*>(&d[i % NPipe][barId * 32 + laneId]), reinterpret_cast<uint32_t*>(in + offsetL));
                cp_mbarrier_arrive(&bar[i % NPipe][barId]);
            }
        }
    }
    const int offset = threadIdx.x + blockIdx.x * THREADS;
    if (offset < N) {
        out[offset] = res;
    }
}

template <int Iters>
void launch_warp_specialize_shared(float* in, float* out, int N, int reps) {
    const int blocks = ( N + THREADS * reps - 1) / (THREADS * reps);
    warp_specialize_shared<2, Iters><<<blocks, THREADS>>>(in, out, N);
    warp_specialize_shared<3, Iters><<<blocks, THREADS>>>(in, out, N);
    warp_specialize_shared<4, Iters><<<blocks, THREADS>>>(in, out, N);
}  

template <int NPipe, int Iters> 
__global__ void warp_specialize_shared_v2(float* __restrict__ in, float* __restrict__ out, int N) {
    __align__(16) __shared__ float d[NPipe][THREADS / 2];
    const int warpId = threadIdx.x / 32;
    const int barId = warpId / 2;
    const bool producer = (warpId % 2 == 0);
    const bool consumer = (warpId % 2 == 1);

    const int laneId = threadIdx.x % 32;
    // auto block = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(block);
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar[NPipe][THREADS / 32 / 2];
    // __shared__ int64_t bar[NPipe][THREADS / 32 / 2];
    if (laneId == 0 && producer) {
        for (int i = 0; i < NPipe; ++i) {
            init(&bar[i][barId], 32);
            // mbarrier_init(&bar[i][barId], 32);
        }
    }
    __syncthreads();

    if (producer){
        for (int p = 0; p < NPipe - 1; ++p) {
            const int offset = barId * 32 + laneId + blockIdx.x * THREADS / 2 + p * gridDim.x * THREADS / 2;
            cuda::memcpy_async(&d[p][barId * 32 + laneId], in + offset, (offset < N) ? sizeof(float) : 0, bar[p][barId]);
            // bar[p][barId].arrive();
            // if (offset < N) {
            //     cp_async_4(reinterpret_cast<uint32_t*>(&d[p][barId * 32 + laneId]), reinterpret_cast<uint32_t*>(in + offset));
            //     cp_mbarrier_arrive(&bar[p][barId]);
            // }
        }
    }
    float res = 0.0f;
    const int iters = divup(N, gridDim.x * THREADS / 2);
    for (int i = NPipe - 1; i < iters + NPipe - 1; i += 1) {
        
        if (consumer) {
            // bar[(i - NPipe + 1) % NPipe][barId].wait();
            // uint64_t token = mbarrier_arrive(&bar[(i - NPipe + 1) % NPipe][barId]);
            // mbarrier_wait(&bar[(i - NPipe + 1) % NPipe][barId], token);
            bar[(i - NPipe + 1) % NPipe][barId].arrive_and_wait();
            res += compute<Iters>(d[(i - NPipe + 1) % NPipe][barId * 32 + laneId]);
        } else if (i < iters) {
            const int offsetL = barId * 32 + laneId + blockIdx.x * THREADS / 2 + i * gridDim.x * THREADS / 2;
            cuda::memcpy_async(&d[i % NPipe][barId * 32 + laneId], in + offsetL, (offsetL < N) ? sizeof(float) : 0, bar[i % NPipe][barId]);
            // bar[i % NPipe][barId].arrive();
            // if (offsetL < N) {
            //     cp_async_4(reinterpret_cast<uint32_t*>(&d[i % NPipe][barId * 32 + laneId]), reinterpret_cast<uint32_t*>(in + offsetL));
            //     cp_mbarrier_arrive(&bar[i % NPipe][barId]);
            // }
        }
    }
    const int offset = threadIdx.x + blockIdx.x * THREADS;
    if (offset < N) {
        out[offset] = res;
    }
}

template <int Iters>
void launch_warp_specialize_shared_v2(float* in, float* out, int N, int reps) {
    const int blocks = ( N + THREADS * reps - 1) / (THREADS * reps);
    warp_specialize_shared_v2<2, Iters><<<blocks, THREADS>>>(in, out, N);
    warp_specialize_shared_v2<3, Iters><<<blocks, THREADS>>>(in, out, N);
    warp_specialize_shared_v2<4, Iters><<<blocks, THREADS>>>(in, out, N);
}  

template <int Iters>
void launch_kernels(float* in, float* out, int N, int reps) {
    launch_baseline<Iters>(in, out, N, reps);
    launch_reg_pipeline<Iters>(in, out, N, reps);
    launch_shared_pipeline<Iters>(in, out, N, reps);
    launch_warp_specialize_reg<Iters>(in, out, N, reps);
    launch_warp_specialize_shared<Iters>(in, out, N, reps);
    launch_warp_specialize_shared_v2<Iters>(in, out, N, reps);
}

// nvcc -lineinfo -O3 -Xptxas -v -o micro micro.cu
int main(int argc, char** argv) {

    int N = 1024 * 1024;
    int reps = 32;

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        reps = atoi(argv[2]);
    }

    float *in, *out;        
    cudaMalloc(&in, N * sizeof(float));
    cudaMalloc(&out, N * sizeof(float));


    launch_kernels<1>(in, out, N, reps);
    launch_kernels<2>(in, out, N, reps);
    launch_kernels<3>(in, out, N, reps);
    launch_kernels<4>(in, out, N, reps);
    // launch_kernels<8>(in, out, N, reps);
    // launch_kernels<16>(in, out, N, reps);
    // launch_kernels<32>(in, out, N, reps);
    // launch_kernels<64>(in, out, N, reps);
    // launch_kernels<128>(in, out, N, reps);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
