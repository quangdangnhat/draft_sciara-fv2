/**
 * sciara_fv2_tiled.cu - CUDA Tiled Version (OPTIMIZED)
 *
 * Uses shared memory for tiling to improve memory access patterns.
 *
 * OPTIMIZATIONS APPLIED:
 * 1. Pointer swapping instead of cudaMemcpy
 * 2. Direct vent update (no full-grid kernel for emitLava)
 * 3. Reduced cudaDeviceSynchronize calls
 * 4. Warp-level reduction with __shfl_down_sync
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
// CUDA configuration - Tile sizes for shared memory
// ----------------------------------------------------------------------------
#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16

// CUDA error checking
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
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)])

// Device constants
__constant__ int d_Xi[MOORE_NEIGHBORS];
__constant__ int d_Xj[MOORE_NEIGHBORS];

int h_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
int h_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

// ----------------------------------------------------------------------------
// CUDA Kernel: computeOutflows (Tiled with shared memory)
// ----------------------------------------------------------------------------
__global__ void kernel_computeOutflows_tiled(
    int r, int c,
    double* __restrict__ Sz,
    double* __restrict__ Sh,
    double* __restrict__ ST,
    double* __restrict__ Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    __shared__ double s_Sz[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    // Load tile into shared memory
    if (i < r && j < c) {
        s_Sz[ty][tx] = GET(Sz, c, i, j);
        s_Sh[ty][tx] = GET(Sh, c, i, j);
        s_ST[ty][tx] = GET(ST, c, i, j);
    } else {
        s_Sz[ty][tx] = -9999.0;
        s_Sh[ty][tx] = 0.0;
        s_ST[ty][tx] = 0.0;
    }
    __syncthreads();

    if (i >= r || j >= c) return;

    double h0 = s_Sh[ty][tx];
    if (h0 <= 0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    double T = s_ST[ty][tx];
    double rr = pow(10.0, _a + _b * T);
    double hc = pow(10.0, _c + _d * T);
    double sz0 = s_Sz[ty][tx];

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k];
        int nj = j + d_Xj[k];
        int local_ni = ty + d_Xi[k];
        int local_nj = tx + d_Xj[k];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            eliminated[k] = true;
            continue;
        }

        double sz, nh;
        if (local_ni >= 0 && local_ni < TILE_SIZE_Y &&
            local_nj >= 0 && local_nj < TILE_SIZE_X) {
            sz = s_Sz[local_ni][local_nj];
            nh = s_Sh[local_ni][local_nj];
        } else {
            sz = GET(Sz, c, ni, nj);
            nh = GET(Sh, c, ni, nj);
        }

        h[k] = nh;
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
// CUDA Kernel: massBalance (Tiled)
// ----------------------------------------------------------------------------
__global__ void kernel_massBalance_tiled(
    int r, int c,
    double* __restrict__ Sh,
    double* __restrict__ Sh_next,
    double* __restrict__ ST,
    double* __restrict__ ST_next,
    double* __restrict__ Mf)
{
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    if (i < r && j < c) {
        s_Sh[ty][tx] = GET(Sh, c, i, j);
        s_ST[ty][tx] = GET(ST, c, i, j);
    } else {
        s_Sh[ty][tx] = 0.0;
        s_ST[ty][tx] = 0.0;
    }
    __syncthreads();

    if (i >= r || j >= c) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = s_Sh[ty][tx];
    double initial_t = s_ST[ty][tx];
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) continue;

        int local_ni = ty + d_Xi[n];
        int local_nj = tx + d_Xj[n];

        double neigh_t;
        if (local_ni >= 0 && local_ni < TILE_SIZE_Y &&
            local_nj >= 0 && local_nj < TILE_SIZE_X) {
            neigh_t = s_ST[local_ni][local_nj];
        } else {
            neigh_t = GET(ST, c, ni, nj);
        }

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
// CUDA Kernel: computeNewTemperatureAndSolidification (Tiled)
// ----------------------------------------------------------------------------
__global__ void kernel_computeNewTemperatureAndSolidification_tiled(
    int r, int c,
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double* __restrict__ Sz, double* __restrict__ Sz_next,
    double* __restrict__ Sh, double* __restrict__ Sh_next,
    double* __restrict__ ST, double* __restrict__ ST_next,
    double* __restrict__ Mhs, bool* __restrict__ Mb)
{
    __shared__ double s_Sz[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ bool s_Mb[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    if (i < r && j < c) {
        s_Sz[ty][tx] = GET(Sz, c, i, j);
        s_Sh[ty][tx] = GET(Sh, c, i, j);
        s_ST[ty][tx] = GET(ST, c, i, j);
        s_Mb[ty][tx] = GET(Mb, c, i, j);
    }
    __syncthreads();

    if (i >= r || j >= c) return;

    double z = s_Sz[ty][tx];
    double h = s_Sh[ty][tx];
    double T = s_ST[ty][tx];

    if (h > 0 && s_Mb[ty][tx] == false) {
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

    // Prepare vent data - store indices for direct access
    int num_vents = sciara->simulation->vent.size();
    int* vent_indices = new int[num_vents];
    for (int k = 0; k < num_vents; k++) {
        int vx = sciara->simulation->vent[k].x();
        int vy = sciara->simulation->vent[k].y();
        vent_indices[k] = vy * c + vx;
    }

    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 gridDim((c + TILE_SIZE_X - 1) / TILE_SIZE_X,
                 (r + TILE_SIZE_Y - 1) / TILE_SIZE_Y);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold)))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // OPTIMIZATION 2: Direct vent update via Unified Memory
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

        // 1. Compute Outflows
        kernel_computeOutflows_tiled<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz, sciara->substates->Sh,
            sciara->substates->ST, sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a, sciara->parameters->b,
            sciara->parameters->c, sciara->parameters->d);

        // 2. Mass Balance
        kernel_massBalance_tiled<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());

        // OPTIMIZATION 1: Pointer swap
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 3. Temperature and Solidification
        kernel_computeNewTemperatureAndSolidification_tiled<<<gridDim, blockDim>>>(
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

        // OPTIMIZATION 1: Pointer swap
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 4. Reduction
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

    delete[] vent_indices;

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
