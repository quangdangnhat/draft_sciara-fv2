/**
 * block_size_exploration.cu - Block Size Exploration for Sciara-fv2
 *
 * This file tests different block configurations to find optimal settings.
 * Compile with: nvcc -O3 -arch=sm_52 block_size_exploration.cu -o block_explore
 *
 * Tests block sizes: 8x8, 16x16, 32x32, 32x8, 16x32
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Grid dimensions (Etna 2006 dataset)
#define ROWS 378
#define COLS 517
#define TOTAL_CELLS (ROWS * COLS)

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Simple test kernel mimicking computeOutflows memory pattern
__global__ void test_kernel_neighbor_access(double* Sz, double* Sh, double* out, int r, int c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    // Simulate neighbor access pattern
    double sum = 0.0;
    int idx = i * c + j;

    // Center
    sum += Sz[idx] + Sh[idx];

    // 8 neighbors (with bounds check)
    int offsets[8][2] = {{-1,0}, {0,-1}, {0,1}, {1,0}, {-1,-1}, {1,-1}, {1,1}, {-1,1}};
    for (int k = 0; k < 8; k++) {
        int ni = i + offsets[k][0];
        int nj = j + offsets[k][1];
        if (ni >= 0 && ni < r && nj >= 0 && nj < c) {
            int nidx = ni * c + nj;
            sum += Sz[nidx] * 0.1 + Sh[nidx] * 0.1;
        }
    }

    out[idx] = sum;
}

// Test kernel for element-wise operations (like solidification)
__global__ void test_kernel_elementwise(double* in, double* out, int r, int c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;
    double val = in[idx];

    // Simulate some computation
    out[idx] = pow(val, 1.0/3.0) * 1.5 + sin(val * 0.01);
}

void run_block_test(int bx, int by, int iterations = 100) {
    double *d_Sz, *d_Sh, *d_out;
    size_t size = TOTAL_CELLS * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_Sz, size));
    CUDA_CHECK(cudaMalloc(&d_Sh, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));

    // Initialize with some data
    CUDA_CHECK(cudaMemset(d_Sz, 0, size));
    CUDA_CHECK(cudaMemset(d_Sh, 0, size));

    dim3 blockDim(bx, by);
    dim3 gridDim((COLS + bx - 1) / bx, (ROWS + by - 1) / by);

    // Warmup
    for (int i = 0; i < 10; i++) {
        test_kernel_neighbor_access<<<gridDim, blockDim>>>(d_Sz, d_Sh, d_out, ROWS, COLS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test neighbor access kernel
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        test_kernel_neighbor_access<<<gridDim, blockDim>>>(d_Sz, d_Sh, d_out, ROWS, COLS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float neighbor_ms;
    CUDA_CHECK(cudaEventElapsedTime(&neighbor_ms, start, stop));

    // Test elementwise kernel
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        test_kernel_elementwise<<<gridDim, blockDim>>>(d_Sz, d_out, ROWS, COLS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elementwise_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elementwise_ms, start, stop));

    // Get occupancy info
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
        test_kernel_neighbor_access, bx * by, 0);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    float theoretical_occupancy = (float)(maxActiveBlocks * bx * by) / 2048.0f; // Max 2048 threads/SM on Maxwell

    printf("| %2dx%-2d | %4d | (%3d,%3d) | %8.3f | %8.3f | %6.2f%% | %d |\n",
           bx, by, bx*by, gridDim.x, gridDim.y,
           neighbor_ms / iterations, elementwise_ms / iterations,
           theoretical_occupancy * 100, maxActiveBlocks);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Sz));
    CUDA_CHECK(cudaFree(d_Sh));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=======================================================================\n");
    printf("Block Size Exploration for Sciara-fv2 (Grid: %dx%d = %d cells)\n", ROWS, COLS, TOTAL_CELLS);
    printf("=======================================================================\n\n");

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Shared memory per block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);

    printf("| Block | Thrds | Grid Size | Neighbor | Elemwise | Occupancy | Blks/SM |\n");
    printf("|-------|-------|-----------|----------|----------|-----------|--------|\n");

    // Test various block configurations
    run_block_test(8, 8);
    run_block_test(16, 8);
    run_block_test(8, 16);
    run_block_test(16, 16);
    run_block_test(32, 8);
    run_block_test(8, 32);
    run_block_test(32, 16);
    run_block_test(16, 32);
    run_block_test(32, 32);

    printf("\n");
    printf("Notes:\n");
    printf("- Neighbor: Time for kernel with 8-neighbor access pattern (like computeOutflows)\n");
    printf("- Elemwise: Time for kernel with element-wise operations (like solidification)\n");
    printf("- Occupancy: Theoretical occupancy based on block size and register usage\n");
    printf("- Blks/SM: Maximum active blocks per SM\n");

    return 0;
}
