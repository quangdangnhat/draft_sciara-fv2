/**
 * sciara_fv2.cu - CUDA Global Memory Version (OPTIMIZED)
 *
 * CUDA implementation using only global memory.
 *
 * OPTIMIZATIONS APPLIED:
 * 1. Pointer swapping instead of cudaMemcpy
 * 2. Direct vent update (no full-grid kernel for emitLava)
 * 3. Reduced cudaDeviceSynchronize calls
 * 4. Optimized block size (32x8 for better coalescing)
 * 5. Warp-level reduction with __shfl_down_sync
 */

#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// CUDA configuration - OPTIMIZED: 32x8 for better memory coalescing
// ----------------------------------------------------------------------------
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ----------------------------------------------------------------------------
// Device macros for array access
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)])

// ----------------------------------------------------------------------------
// Device constants for neighborhood
// ----------------------------------------------------------------------------
__constant__ int d_Xi[MOORE_NEIGHBORS];
__constant__ int d_Xj[MOORE_NEIGHBORS];

// Host arrays for neighborhood (same as original)
int h_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
int h_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

// ----------------------------------------------------------------------------
// CUDA Kernel: computeOutflows
// ----------------------------------------------------------------------------
__global__ void kernel_computeOutflows(
    int r, int c,
    double* __restrict__ Sz,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    double* __restrict__ Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    // Skip if no lava
    double h0 = GET(Sh, c, i, j);
    if (h0 <= 0) return;

    // Local arrays
    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    double T = GET(ST, c, i, j);
    double rr = pow(10.0, _a + _b * T);
    double hc = pow(10.0, _c + _d * T);

    double sz0 = GET(Sz, c, i, j);

    // Initialize neighbor data
    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k];
        int nj = j + d_Xj[k];

        // Bounds check
        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            eliminated[k] = true;
            continue;
        }

        double sz = GET(Sz, c, ni, nj);
        h[k] = GET(Sh, c, ni, nj);
        w[k] = Pc;
        Pr[k] = rr;

        // Diagonal correction for effective height
        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);

        eliminated[k] = false;
    }

    // Initialize H and theta
    H[0] = z[0];
    theta[0] = 0;
    eliminated[0] = false;

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (eliminated[k]) continue;

        if (z[0] + h[0] > z[k] + h[k]) {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
        } else {
            eliminated[k] = true;
        }
    }

    // Minimization algorithm
    double avg;
    int counter;
    bool loop;
    do {
        loop = false;
        avg = h[0];
        counter = 0;

        for (int k = 0; k < MOORE_NEIGHBORS; k++) {
            if (!eliminated[k]) {
                avg += H[k];
                counter++;
            }
        }

        if (counter != 0)
            avg = avg / (double)counter;

        for (int k = 0; k < MOORE_NEIGHBORS; k++) {
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k] = true;
                loop = true;
            }
        }
    } while (loop);

    // Compute outflows
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        double flow;
        if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
            flow = Pr[k] * (avg - H[k]);
        } else {
            flow = 0.0;
        }
        BUF_SET(Mf, r, c, k - 1, i, j, flow);
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: massBalance
// ----------------------------------------------------------------------------
__global__ void kernel_massBalance(
    int r, int c,
    double* __restrict__ Sh,
    double* __restrict__ Sh_next,
    double* __restrict__ ST,
    double* __restrict__ ST_next,
    double* __restrict__ Mf)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    // Inflow indices mapping (opposite directions)
    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = GET(Sh, c, i, j);
    double initial_t = GET(ST, c, i, j);
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        // Bounds check
        if (ni < 0 || ni >= r || nj < 0 || nj >= c) continue;

        double neigh_t = GET(ST, c, ni, nj);
        double inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], ni, nj);
        double outFlow = BUF_GET(Mf, r, c, n - 1, i, j);

        h_next += inFlow - outFlow;
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) {
        t_next /= h_next;
        SET(ST_next, c, i, j, t_next);
        SET(Sh_next, c, i, j, h_next);
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: computeNewTemperatureAndSolidification
// ----------------------------------------------------------------------------
__global__ void kernel_computeNewTemperatureAndSolidification(
    int r, int c,
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double* __restrict__ Sz, double* __restrict__ Sz_next,
    double* __restrict__ Sh, double* __restrict__ Sh_next,
    double* __restrict__ ST, double* __restrict__ ST_next,
    double* __restrict__ Mhs, bool* __restrict__ Mb)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double z = GET(Sz, c, i, j);
    double h = GET(Sh, c, i, j);
    double T = GET(ST, c, i, j);

    if (h > 0 && GET(Mb, c, i, j) == false) {
        double aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
        double nT = T / pow(aus, 1.0 / 3.0);

        if (nT > PTsol) {
            // No solidification
            SET(ST_next, c, i, j, nT);
        } else {
            // Solidification
            SET(Sz_next, c, i, j, z + h);
            SET(Sh_next, c, i, j, 0.0);
            SET(ST_next, c, i, j, PTsol);
            SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: Optimized Reduction with warp shuffle
// ----------------------------------------------------------------------------
__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kernel_reduceAdd(double* input, double* output, int size)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and do first reduction during load
    double sum = 0.0;
    if (i < size) sum += input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];

    // Warp-level reduction first
    sum = warpReduceSum(sum);

    // Write warp results to shared memory
    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) sdata[warpId] = sum;
    __syncthreads();

    // Final reduction by first warp
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < numWarps) {
        sum = sdata[tid];
    } else {
        sum = 0.0;
    }

    if (warpId == 0) {
        sum = warpReduceSum(sum);
        if (tid == 0) output[blockIdx.x] = sum;
    }
}

// Host function for reduction
double reduceAddCUDA(double* d_buffer, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);

    double* d_partial;
    CUDA_CHECK(cudaMallocManaged(&d_partial, numBlocks * sizeof(double)));

    int sharedSize = (blockSize / 32 + 1) * sizeof(double);
    kernel_reduceAdd<<<numBlocks, blockSize, sharedSize>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Continue reducing until single value
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        kernel_reduceAdd<<<newNumBlocks, blockSize, sharedSize>>>(d_partial, d_partial, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        numBlocks = newNumBlocks;
    }

    double result = d_partial[0];
    CUDA_CHECK(cudaFree(d_partial));

    return result;
}

// ----------------------------------------------------------------------------
// OPTIMIZATION: Copy Sh/ST to next buffers for kernels that need both
// ----------------------------------------------------------------------------
__global__ void kernel_copyBuffers(int r, int c,
    double* __restrict__ src1, double* __restrict__ dst1,
    double* __restrict__ src2, double* __restrict__ dst2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    int idx = i * c + j;
    dst1[idx] = src1[idx];
    dst2[idx] = src2[idx];
}

// ----------------------------------------------------------------------------
// Main function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    // Input data
    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    // Domain dimensions
    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    // Copy neighborhood to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xi, h_Xi, MOORE_NEIGHBORS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xj, h_Xj, MOORE_NEIGHBORS * sizeof(int)));

    // Simulation initialization
    double total_current_lava = -1;
    simulationInitialize(sciara);

    // Prepare vent data - store indices for direct access
    int num_vents = sciara->simulation->vent.size();
    int* vent_indices = new int[num_vents];
    for (int k = 0; k < num_vents; k++) {
        int vx = sciara->simulation->vent[k].x();
        int vy = sciara->simulation->vent[k].y();
        vent_indices[k] = vy * c + vx;
    }

    // CUDA grid/block configuration - OPTIMIZED 32x8
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((c + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                 (r + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    // Main simulation loop
    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold)))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // OPTIMIZATION 2: Direct vent update (no kernel launch for few vents)
        // Update vent thickness and emit lava directly via Unified Memory
        for (int k = 0; k < num_vents; k++) {
            double thickness = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time,
                sciara->parameters->Pclock,
                sciara->simulation->emission_time,
                sciara->parameters->Pac);

            sciara->simulation->total_emitted_lava += thickness;

            int idx = vent_indices[k];
            sciara->substates->Sh[idx] += thickness;
            sciara->substates->ST[idx] = sciara->parameters->PTvent;
        }

        // 1. Compute Outflows (reads Sh, ST, Sz; writes Mf)
        kernel_computeOutflows<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz,
            sciara->substates->Sh,
            sciara->substates->ST,
            sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a,
            sciara->parameters->b,
            sciara->parameters->c,
            sciara->parameters->d);
        // No sync needed - next kernel depends on Mf which is written here

        // 2. Mass Balance (reads Sh, ST, Mf; writes Sh_next, ST_next)
        kernel_massBalance<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());

        // OPTIMIZATION 1: Pointer swap instead of cudaMemcpy
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 3. Temperature and Solidification
        kernel_computeNewTemperatureAndSolidification<<<gridDim, blockDim>>>(
            r, c,
            sciara->parameters->Pepsilon,
            sciara->parameters->Psigma,
            sciara->parameters->Pclock,
            sciara->parameters->Pcool,
            sciara->parameters->Prho,
            sciara->parameters->Pcv,
            sciara->parameters->Pac,
            sciara->parameters->PTsol,
            sciara->substates->Sz, sciara->substates->Sz_next,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mhs, sciara->substates->Mb);
        CUDA_CHECK(cudaDeviceSynchronize());

        // OPTIMIZATION 1: Pointer swap instead of cudaMemcpy
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 4. Global reduction (periodically)
        if (sciara->simulation->step % reduceInterval == 0) {
            total_current_lava = reduceAddCUDA(sciara->substates->Sh, r * c);
        }
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    // Cleanup
    delete[] vent_indices;

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
