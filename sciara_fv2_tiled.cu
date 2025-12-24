/**
 * sciara_fv2_tiled.cu - CUDA Tiled Version (Shared Memory, No Halo)
 *
 * Uses shared memory for tiling within each block.
 * Neighbor accesses at tile boundaries still go to global memory.
 *
 * Kernel order (as per Sciara-fv2 model):
 * 1. emitLava
 * 2. computeOutflows
 * 3. massBalance
 * 4. computeNewTemperatureAndSolidification
 * 5. boundaryConditions
 */

#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

#define TILE_SIZE_X 16
#define TILE_SIZE_Y 16

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[((n)*(rows)*(columns)) + ((i)*(columns)) + (j)])

__constant__ int d_Xi[MOORE_NEIGHBORS];
__constant__ int d_Xj[MOORE_NEIGHBORS];
int h_Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
int h_Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

// ----------------------------------------------------------------------------
// Kernel: emitLava
// ----------------------------------------------------------------------------
__global__ void kernel_emitLava(
    int r, int c,
    int* vent_x, int* vent_y, double* vent_thickness,
    int num_vents, double PTvent,
    double* Sh, double* Sh_next, double* ST, double* ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= r || j >= c) return;

    for (int k = 0; k < num_vents; k++) {
        if (i == vent_y[k] && j == vent_x[k]) {
            SET(Sh_next, c, i, j, GET(Sh, c, i, j) + vent_thickness[k]);
            SET(ST_next, c, i, j, PTvent);
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel: computeOutflows (Tiled)
// ----------------------------------------------------------------------------
__global__ void kernel_computeOutflows_tiled(
    int r, int c, double* Sz, double* Sh, double* ST, double* Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    __shared__ double s_Sz[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    if (i < r && j < c) {
        s_Sz[ty][tx] = GET(Sz, c, i, j);
        s_Sh[ty][tx] = GET(Sh, c, i, j);
        s_ST[ty][tx] = GET(ST, c, i, j);
    } else {
        s_Sz[ty][tx] = -9999.0; s_Sh[ty][tx] = 0.0; s_ST[ty][tx] = 0.0;
    }
    __syncthreads();

    if (i >= r || j >= c) return;
    double h0 = s_Sh[ty][tx];
    if (h0 <= 0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS], h[MOORE_NEIGHBORS], H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS], w[MOORE_NEIGHBORS], Pr[MOORE_NEIGHBORS];

    double T = s_ST[ty][tx];
    double rr = pow(10.0, _a + _b * T);
    double hc = pow(10.0, _c + _d * T);
    double sz0 = s_Sz[ty][tx];

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k], nj = j + d_Xj[k];
        int lni = ty + d_Xi[k], lnj = tx + d_Xj[k];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) { eliminated[k] = true; continue; }

        double sz, nh;
        if (lni >= 0 && lni < TILE_SIZE_Y && lnj >= 0 && lnj < TILE_SIZE_X) {
            sz = s_Sz[lni][lnj]; nh = s_Sh[lni][lnj];
        } else {
            sz = GET(Sz, c, ni, nj); nh = GET(Sh, c, ni, nj);
        }
        h[k] = nh; w[k] = Pc; Pr[k] = rr;
        z[k] = (k < VON_NEUMANN_NEIGHBORS) ? sz : sz0 - (sz0 - sz) / sqrt(2.0);
        eliminated[k] = false;
    }

    H[0] = z[0]; theta[0] = 0; eliminated[0] = false;
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (eliminated[k]) continue;
        if (z[0] + h[0] > z[k] + h[k]) {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
        } else eliminated[k] = true;
    }

    double avg; int counter; bool loop;
    do {
        loop = false; avg = h[0]; counter = 0;
        for (int k = 0; k < MOORE_NEIGHBORS; k++) if (!eliminated[k]) { avg += H[k]; counter++; }
        if (counter != 0) avg /= (double)counter;
        for (int k = 0; k < MOORE_NEIGHBORS; k++) if (!eliminated[k] && avg <= H[k]) { eliminated[k] = true; loop = true; }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        double flow = (!eliminated[k] && h[0] > hc * cos(theta[k])) ? Pr[k] * (avg - H[k]) : 0.0;
        BUF_SET(Mf, r, c, k - 1, i, j, flow);
    }
}

// ----------------------------------------------------------------------------
// Kernel: massBalance (Tiled)
// ----------------------------------------------------------------------------
__global__ void kernel_massBalance_tiled(
    int r, int c, double* Sh, double* Sh_next, double* ST, double* ST_next, double* Mf)
{
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    if (i < r && j < c) { s_Sh[ty][tx] = GET(Sh, c, i, j); s_ST[ty][tx] = GET(ST, c, i, j); }
    else { s_Sh[ty][tx] = 0.0; s_ST[ty][tx] = 0.0; }
    __syncthreads();

    if (i >= r || j >= c) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
    double initial_h = s_Sh[ty][tx], initial_t = s_ST[ty][tx];
    double h_next = initial_h, t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n], nj = j + d_Xj[n];
        if (ni < 0 || ni >= r || nj < 0 || nj >= c) continue;
        int lni = ty + d_Xi[n], lnj = tx + d_Xj[n];
        double neigh_t = (lni >= 0 && lni < TILE_SIZE_Y && lnj >= 0 && lnj < TILE_SIZE_X) ? s_ST[lni][lnj] : GET(ST, c, ni, nj);
        double inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], ni, nj);
        double outFlow = BUF_GET(Mf, r, c, n - 1, i, j);
        h_next += inFlow - outFlow;
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) { SET(ST_next, c, i, j, t_next / h_next); SET(Sh_next, c, i, j, h_next); }
}

// ----------------------------------------------------------------------------
// Kernel: computeNewTemperatureAndSolidification (Tiled)
// ----------------------------------------------------------------------------
__global__ void kernel_computeNewTemperatureAndSolidification_tiled(
    int r, int c, double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double* Sz, double* Sz_next, double* Sh, double* Sh_next,
    double* ST, double* ST_next, double* Mhs, bool* Mb)
{
    __shared__ double s_Sz[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_Sh[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ double s_ST[TILE_SIZE_Y][TILE_SIZE_X];
    __shared__ bool s_Mb[TILE_SIZE_Y][TILE_SIZE_X];

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE_SIZE_X + tx;
    int i = blockIdx.y * TILE_SIZE_Y + ty;

    if (i < r && j < c) {
        s_Sz[ty][tx] = GET(Sz, c, i, j); s_Sh[ty][tx] = GET(Sh, c, i, j);
        s_ST[ty][tx] = GET(ST, c, i, j); s_Mb[ty][tx] = GET(Mb, c, i, j);
    }
    __syncthreads();

    if (i >= r || j >= c) return;
    double z = s_Sz[ty][tx], h = s_Sh[ty][tx], T = s_ST[ty][tx];

    if (h > 0 && s_Mb[ty][tx] == false) {
        double aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
        double nT = T / pow(aus, 1.0 / 3.0);
        if (nT > PTsol) { SET(ST_next, c, i, j, nT); }
        else {
            SET(Sz_next, c, i, j, z + h); SET(Sh_next, c, i, j, 0.0);
            SET(ST_next, c, i, j, PTsol); SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel: boundaryConditions
// ----------------------------------------------------------------------------
__global__ void kernel_boundaryConditions(
    int r, int c, bool* Mb, double* Sh, double* Sh_next, double* ST, double* ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= r || j >= c) return;
    return; // Disabled in original
    if (GET(Mb, c, i, j)) { SET(Sh_next, c, i, j, 0.0); SET(ST_next, c, i, j, 0.0); }
}

// ----------------------------------------------------------------------------
// Reduction
// ----------------------------------------------------------------------------
__inline__ __device__ double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kernel_reduceAdd(double* input, double* output, int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    double sum = 0.0;
    if (i < size) sum += input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];
    sum = warpReduceSum(sum);
    int lane = tid % warpSize, warpId = tid / warpSize;
    if (lane == 0) sdata[warpId] = sum;
    __syncthreads();
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    sum = (tid < numWarps) ? sdata[tid] : 0.0;
    if (warpId == 0) { sum = warpReduceSum(sum); if (tid == 0) output[blockIdx.x] = sum; }
}

double reduceAddCUDA(double* d_buffer, int size) {
    int blockSize = 256, numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);
    double* d_partial; CUDA_CHECK(cudaMallocManaged(&d_partial, numBlocks * sizeof(double)));
    int sharedSize = (blockSize / 32 + 1) * sizeof(double);
    kernel_reduceAdd<<<numBlocks, blockSize, sharedSize>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        kernel_reduceAdd<<<newNumBlocks, blockSize, sharedSize>>>(d_partial, d_partial, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize()); numBlocks = newNumBlocks;
    }
    double result = d_partial[0]; CUDA_CHECK(cudaFree(d_partial)); return result;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    Sciara *sciara; init(sciara);
    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);
    int r = sciara->domain->rows, c = sciara->domain->cols;

    CUDA_CHECK(cudaMemcpyToSymbol(d_Xi, h_Xi, MOORE_NEIGHBORS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xj, h_Xj, MOORE_NEIGHBORS * sizeof(int)));

    double total_current_lava = -1;
    simulationInitialize(sciara);

    int num_vents = sciara->simulation->vent.size();
    int *d_vent_x, *d_vent_y; double *d_vent_thickness;
    CUDA_CHECK(cudaMallocManaged(&d_vent_x, num_vents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_vent_y, num_vents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_vent_thickness, num_vents * sizeof(double)));
    for (int k = 0; k < num_vents; k++) {
        d_vent_x[k] = sciara->simulation->vent[k].x();
        d_vent_y[k] = sciara->simulation->vent[k].y();
    }

    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 gridDim((c + TILE_SIZE_X - 1) / TILE_SIZE_X, (r + TILE_SIZE_Y - 1) / TILE_SIZE_Y);

    util::Timer cl_timer;
    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold))) {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // Sync before CPU writes to d_vent_thickness (UVM requires this)
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int k = 0; k < num_vents; k++) {
            d_vent_thickness[k] = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time, sciara->parameters->Pclock,
                sciara->simulation->emission_time, sciara->parameters->Pac);
            sciara->simulation->total_emitted_lava += d_vent_thickness[k];
        }

        // 1. emitLava
        kernel_emitLava<<<gridDim, blockDim>>>(r, c, d_vent_x, d_vent_y, d_vent_thickness,
            num_vents, sciara->parameters->PTvent, sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 2. computeOutflows
        kernel_computeOutflows_tiled<<<gridDim, blockDim>>>(r, c, sciara->substates->Sz,
            sciara->substates->Sh, sciara->substates->ST, sciara->substates->Mf,
            sciara->parameters->Pc, sciara->parameters->a, sciara->parameters->b,
            sciara->parameters->c, sciara->parameters->d);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. massBalance
        kernel_massBalance_tiled<<<gridDim, blockDim>>>(r, c, sciara->substates->Sh,
            sciara->substates->Sh_next, sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 4. computeNewTemperatureAndSolidification
        kernel_computeNewTemperatureAndSolidification_tiled<<<gridDim, blockDim>>>(r, c,
            sciara->parameters->Pepsilon, sciara->parameters->Psigma, sciara->parameters->Pclock,
            sciara->parameters->Pcool, sciara->parameters->Prho, sciara->parameters->Pcv,
            sciara->parameters->Pac, sciara->parameters->PTsol, sciara->substates->Sz,
            sciara->substates->Sz_next, sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next, sciara->substates->Mhs,
            sciara->substates->Mb);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(sciara->substates->Sz, sciara->substates->Sz_next);
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 5. boundaryConditions
        kernel_boundaryConditions<<<gridDim, blockDim>>>(r, c, sciara->substates->Mb,
            sciara->substates->Sh, sciara->substates->Sh_next, sciara->substates->ST,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(sciara->substates->Sh, sciara->substates->Sh_next);
        std::swap(sciara->substates->ST, sciara->substates->ST_next);

        // 6. Reduction
        if (sciara->simulation->step % reduceInterval == 0)
            total_current_lava = reduceAddCUDA(sciara->substates->Sh, r * c);
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    CUDA_CHECK(cudaFree(d_vent_x)); CUDA_CHECK(cudaFree(d_vent_y)); CUDA_CHECK(cudaFree(d_vent_thickness));
    printf("Releasing memory...\n"); finalize(sciara);
    return 0;
}
