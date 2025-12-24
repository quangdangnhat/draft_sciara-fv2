/**
 * sciara_fv2_cfamo.cu - CUDA CfAMo (Conflict-Free Memory-Optimized) (OPTIMIZED)
 *
 * CfAMo further optimizes CfAMe by eliminating the Mf buffer.
 *
 * KERNELS (in order):
 * 1. emitLava
 * 2. computeOutflows (combined in CfA_Mo)
 * 3. massBalance (combined in CfA_Mo)
 * 4. computeNewTemperatureAndSolidification
 * 5. boundaryConditions
 *
 * OPTIMIZATIONS APPLIED:
 * 1. Pointer swapping instead of cudaMemcpy
 * 2. Reduced cudaDeviceSynchronize calls
 * 3. Warp-level reduction with __shfl_down_sync
 * 4. Optimized block size (32x8)
 */

#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

// ----------------------------------------------------------------------------
// I/O parameters
// ----------------------------------------------------------------------------
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// CUDA configuration - OPTIMIZED
// ----------------------------------------------------------------------------
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

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
// Device macros
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])

// Device constants
__constant__ int d_Xi[MOORE_NEIGHBORS];
__constant__ int d_Xj[MOORE_NEIGHBORS];

int h_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
int h_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

// ----------------------------------------------------------------------------
// Double-precision atomic add
// ----------------------------------------------------------------------------
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// ----------------------------------------------------------------------------
// CUDA Kernel 1: emitLava
// ----------------------------------------------------------------------------
__global__ void kernel_emitLava(
    int r, int c,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    int* __restrict__ vent_indices,
    double* __restrict__ emission_thicknesses,
    double vent_temperature,
    int num_vents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vents) return;

    int cell_idx = vent_indices[idx];
    double thickness = emission_thicknesses[idx];

    if (thickness > 0) {
        Sh[cell_idx] += thickness;
        ST[cell_idx] = vent_temperature;
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel 5: boundaryConditions
// Matches original serial behavior (currently disabled)
// ----------------------------------------------------------------------------
__global__ void kernel_boundaryConditions(
    int r, int c,
    double* __restrict__ Sz,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    bool* __restrict__ Mb)
{
    // In the original serial version, this function does nothing
    // (the boundary is set once at initialization via makeBorder)
    return;
}

// ----------------------------------------------------------------------------
// CUDA Kernel: Initialize buffers for CfAMo step
// ----------------------------------------------------------------------------
__global__ void kernel_initBuffers(
    int r, int c,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    double* __restrict__ Sh_next,
    double* __restrict__ ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double sh = GET(Sh, c, i, j);
    double st = GET(ST, c, i, j);

    SET(Sh_next, c, i, j, sh);
    SET(ST_next, c, i, j, sh * st);
}

// ----------------------------------------------------------------------------
// CUDA Kernel: CfA_Mo - Memory Optimized Combined Kernel
// No Mf buffer needed
// ----------------------------------------------------------------------------
__global__ void kernel_CfA_Mo(
    int r, int c,
    double* __restrict__ Sz,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    double* __restrict__ Sh_next,
    double* __restrict__ ST_next,
    double Pc, double _a, double _b, double _c, double _d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double h0 = GET(Sh, c, i, j);
    if (h0 <= 0) return;

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

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k];
        int nj = j + d_Xj[k];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            eliminated[k] = true;
            continue;
        }

        double sz = GET(Sz, c, ni, nj);
        h[k] = GET(Sh, c, ni, nj);
        w[k] = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);

        eliminated[k] = false;
    }

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
        if (counter != 0) avg = avg / (double)counter;
        for (int k = 0; k < MOORE_NEIGHBORS; k++) {
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k] = true;
                loop = true;
            }
        }
    } while (loop);

    // Compute and apply flows directly
    double total_outflow = 0.0;

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
            double flow = Pr[k] * (avg - H[k]);

            if (flow > 0) {
                int ni = i + d_Xi[k];
                int nj = j + d_Xj[k];

                atomicAddDouble(&Sh_next[ni * c + nj], flow);
                atomicAddDouble(&ST_next[ni * c + nj], flow * T);

                total_outflow += flow;
            }
        }
    }

    if (total_outflow > 0) {
        atomicAddDouble(&Sh_next[i * c + j], -total_outflow);
        atomicAddDouble(&ST_next[i * c + j], -total_outflow * T);
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: Normalize temperature
// ----------------------------------------------------------------------------
__global__ void kernel_normalizeTemperature(
    int r, int c,
    double* __restrict__ Sh_next,
    double* __restrict__ ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double h = GET(Sh_next, c, i, j);
    if (h > 0) {
        double hT = GET(ST_next, c, i, j);
        SET(ST_next, c, i, j, hT / h);
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
            SET(ST_next, c, i, j, nT);
        } else {
            SET(Sz_next, c, i, j, z + h);
            SET(Sh_next, c, i, j, 0.0);
            SET(ST_next, c, i, j, PTsol);
            SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
        }
    }
}

// ----------------------------------------------------------------------------
// Optimized Reduction with warp shuffle
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

    double sum = 0.0;
    if (i < size) sum += input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];

    sum = warpReduceSum(sum);

    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) sdata[warpId] = sum;
    __syncthreads();

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

double reduceAddCUDA(double* d_buffer, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);

    double* d_partial;
    CUDA_CHECK(cudaMallocManaged(&d_partial, numBlocks * sizeof(double)));

    int sharedSize = (blockSize / 32 + 1) * sizeof(double);
    kernel_reduceAdd<<<numBlocks, blockSize, sharedSize>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());

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
// Main function
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    int r = sciara->domain->rows;
    int c = sciara->domain->cols;

    CUDA_CHECK(cudaMemcpyToSymbol(d_Xi, h_Xi, MOORE_NEIGHBORS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xj, h_Xj, MOORE_NEIGHBORS * sizeof(int)));

    double total_current_lava = -1;
    simulationInitialize(sciara);

    // Prepare vent data for GPU - using Unified Memory
    int num_vents = sciara->simulation->vent.size();
    int* d_vent_indices;
    double* d_emission_thicknesses;
    CUDA_CHECK(cudaMallocManaged(&d_vent_indices, num_vents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_emission_thicknesses, num_vents * sizeof(double)));

    for (int k = 0; k < num_vents; k++) {
        int vx = sciara->simulation->vent[k].x();
        int vy = sciara->simulation->vent[k].y();
        d_vent_indices[k] = vy * c + vx;
    }

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((c + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                 (r + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold)))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // =========================================================
        // Kernel 1: emitLava - Lava emission from vents
        // =========================================================
        // Sync before CPU writes to d_emission_thicknesses (UVM requires this)
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int k = 0; k < num_vents; k++) {
            double thickness = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time,
                sciara->parameters->Pclock,
                sciara->simulation->emission_time,
                sciara->parameters->Pac);
            d_emission_thicknesses[k] = thickness;
            sciara->simulation->total_emitted_lava += thickness;
        }
        int ventBlocks = (num_vents + 255) / 256;
        kernel_emitLava<<<ventBlocks, 256>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->ST,
            d_vent_indices, d_emission_thicknesses,
            sciara->parameters->PTvent, num_vents);

        // Initialize buffers for CfAMo
        kernel_initBuffers<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->ST,
            sciara->substates->Sh_next, sciara->substates->ST_next);

        // =========================================================
        // Kernel 2+3: CfA_Mo - Combined computeOutflows + massBalance
        // =========================================================
        kernel_CfA_Mo<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz,
            sciara->substates->Sh,
            sciara->substates->ST,
            sciara->substates->Sh_next,
            sciara->substates->ST_next,
            sciara->parameters->Pc,
            sciara->parameters->a, sciara->parameters->b,
            sciara->parameters->c, sciara->parameters->d);

        // Normalize temperature
        kernel_normalizeTemperature<<<gridDim, blockDim>>>(
            r, c, sciara->substates->Sh_next, sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        // OPTIMIZATION: Pointer swap
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // =========================================================
        // Kernel 4: computeNewTemperatureAndSolidification
        // =========================================================
        kernel_computeNewTemperatureAndSolidification<<<gridDim, blockDim>>>(
            r, c,
            sciara->parameters->Pepsilon, sciara->parameters->Psigma,
            sciara->parameters->Pclock, sciara->parameters->Pcool,
            sciara->parameters->Prho, sciara->parameters->Pcv,
            sciara->parameters->Pac, sciara->parameters->PTsol,
            sciara->substates->Sz, sciara->substates->Sz_next,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mhs, sciara->substates->Mb);
        CUDA_CHECK(cudaDeviceSynchronize());

        // OPTIMIZATION: Pointer swap
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // =========================================================
        // Kernel 5: boundaryConditions - Apply boundary conditions
        // =========================================================
        kernel_boundaryConditions<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz, sciara->substates->Sh,
            sciara->substates->ST, sciara->substates->Mb);

        // Reduction for stopping criterion
        if (sciara->simulation->step % reduceInterval == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
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

    CUDA_CHECK(cudaFree(d_vent_indices));
    CUDA_CHECK(cudaFree(d_emission_thicknesses));

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
