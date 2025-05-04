#include <cooperative_groups.h>
#include <cuda/barrier>
#include <stdio.h>

// __global__ void kernel(float* p) {
//     __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
//     if (threadIdx.x == 0) {
//         init(&barrier, 128);
//     }
//     __shared__ float shared_data[128];
//     cuda::memcpy_async(shared_data + threadIdx.x, p + threadIdx.x, sizeof(float), barrier);
//     barrier.arrive_and_wait();
    
//     p[threadIdx.x] = shared_data[threadIdx.x] + 1.0f;
// }

// dump to ptx
// nvcc -arch=sm_86 -ptx test.cu -o test.ptx

// get sass
// nvcc -arch=sm_86 -cubin test.cu -o test.cubin
// cuobjdump -sass test.cubin > test.sass
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

__global__ void kernel1(int* p, int* q) {
    __shared__ int shared_data[128];
    auto thread_block = cg::this_thread_block();
    auto tile = cg::tiled_partition<128>(thread_block);
    unsigned int val = cg::inclusive_scan(tile, p[threadIdx.x]);
    shared_data[threadIdx.x] = val;
    // val = cg::exclusive_scan(tile, val);
    q[threadIdx.x] = shared_data[threadIdx.x];
}


int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    
    printf("pagable memory accessible: %d\n", deviceProp.pageableMemoryAccess);
    printf("host native atomic: %d\n", deviceProp.hostNativeAtomicSupported);
    printf("pagable memory access uses host page tables : %d\n", deviceProp.pageableMemoryAccessUsesHostPageTables);
    printf("direct manged memory access: %d\n", deviceProp.directManagedMemAccessFromHost);

    printf("managed memory access: %d\n", deviceProp.managedMemory);
    printf("concurrent memory access: %d\n", deviceProp.concurrentManagedAccess);
}
