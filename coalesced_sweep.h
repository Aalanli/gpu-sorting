#include <stdint.h>

void launch_compute_histograms_b32(
    const uint32_t* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int threads, const int radix_log
);

void launch_compute_histograms_pipelined_b32(
    const uint32_t* data, uint32_t* hist, const int N, const bool hist_init, const int nreps_per_thread,
    const int npipe
);

void launch_single_radix_pass_b32(
    const uint32_t* key, // [N]
    const uint32_t* global_histogram, // [RADIX]
    uint32_t* res, // [N]
    uint32_t* hist, // [(NBLOCKS + 1) * RADIX]
    const int N,
    const int bits_offset,
    int* index // [1]
);


void launch_compute_histograms_shared_b32(
    const uint32_t* data, uint32_t* hist, const int N, const int radix_stride, const int nreps_per_thread,
    const int threads, const int radix_dup
);

void launch_compute_histograms_shared_pipelined_b32(
    const uint32_t* data, uint32_t* hist, const int N, const int radix_stride, 
    const int threads,
    const int nreps_per_thread,
    const int radix_dup, const int npipe
);