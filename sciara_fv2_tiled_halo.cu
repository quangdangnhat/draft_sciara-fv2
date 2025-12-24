/**
 * sciara_fv2_tiled_halo.cu - CUDA Tiled Version WITH Halo
 *
 * Uses shared memory with halo regions for complete neighbor access.
 * Each tile loads extra border cells (halo) to eliminate global memory
 * accesses for neighbor data.
 *
 * Tile layout: [HALO + TILE + HALO] in both dimensions
 */

#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// I/O parameters
// ----------------------------------------------------------------------------
#define INPUT_PATH_ID          1
#define OUTPUT_PATH_ID         2
#define MAX_STEPS_ID           3
#define REDUCE_INTERVL_ID      4
#define THICKNESS_THRESHOLD_ID 5

// ----------------------------------------------------------------------------
// CUDA configuration - Tile sizes with halo
// ----------------------------------------------------------------------------
#define TILE_SIZE 16
#define HALO_SIZE 1  // Moore neighborhood requires 1-cell halo
#define BLOCK_SIZE (TILE_SIZE + 2 * HALO_SIZE)  // 18x18 for 16x16 tile

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
// Helper: Load tile with halo into shared memory
// ----------------------------------------------------------------------------
__device__ void loadTileWithHalo(
    double* global, double shared[][BLOCK_SIZE],
    int r, int c, int blockStartI, int blockStartJ,
    int tx, int ty, double defaultVal)
{
    // Each thread may need to load multiple elements to cover halo
    // We use a 2D thread block of TILE_SIZE x TILE_SIZE
    // Need to load BLOCK_SIZE x BLOCK_SIZE elements

    int elementsPerThread = (BLOCK_SIZE * BLOCK_SIZE + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    int tid = ty * TILE_SIZE + tx;

    for (int e = 0; e < elementsPerThread; e++) {
        int linearIdx = tid + e * (TILE_SIZE * TILE_SIZE);
        if (linearIdx < BLOCK_SIZE * BLOCK_SIZE) {
            int si = linearIdx / BLOCK_SIZE;
            int sj = linearIdx % BLOCK_SIZE;

            int gi = blockStartI - HALO_SIZE + si;
            int gj = blockStartJ - HALO_SIZE + sj;

            if (gi >= 0 && gi < r && gj >= 0 && gj < c) {
                shared[si][sj] = GET(global, c, gi, gj);
            } else {
                shared[si][sj] = defaultVal;
            }
        }
    }
}

__device__ void loadTileWithHaloBool(
    bool* global, bool shared[][BLOCK_SIZE],
    int r, int c, int blockStartI, int blockStartJ,
    int tx, int ty, bool defaultVal)
{
    int elementsPerThread = (BLOCK_SIZE * BLOCK_SIZE + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    int tid = ty * TILE_SIZE + tx;

    for (int e = 0; e < elementsPerThread; e++) {
        int linearIdx = tid + e * (TILE_SIZE * TILE_SIZE);
        if (linearIdx < BLOCK_SIZE * BLOCK_SIZE) {
            int si = linearIdx / BLOCK_SIZE;
            int sj = linearIdx % BLOCK_SIZE;

            int gi = blockStartI - HALO_SIZE + si;
            int gj = blockStartJ - HALO_SIZE + sj;

            if (gi >= 0 && gi < r && gj >= 0 && gj < c) {
                shared[si][sj] = GET(global, c, gi, gj);
            } else {
                shared[si][sj] = defaultVal;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: emitLava (Halo version)
// ----------------------------------------------------------------------------
__global__ void kernel_emitLava_halo(
    int r, int c,
    int* vent_x, int* vent_y, double* vent_thickness,
    int num_vents,
    double PTvent,
    double* Sh, double* Sh_next,
    double* ST_next)
{
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;
    int i = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (i >= r || j >= c) return;

    for (int k = 0; k < num_vents; k++) {
        if (i == vent_y[k] && j == vent_x[k]) {
            double current_h = GET(Sh, c, i, j);
            SET(Sh_next, c, i, j, current_h + vent_thickness[k]);
            SET(ST_next, c, i, j, PTvent);
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: computeOutflows WITH Halo
// All neighbor accesses from shared memory
// ----------------------------------------------------------------------------
__global__ void kernel_computeOutflows_halo(
    int r, int c,
    double* Sz, double* Sh, double* ST, double* Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    __shared__ double s_Sz[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_Sh[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_ST[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockStartI = blockIdx.y * TILE_SIZE;
    int blockStartJ = blockIdx.x * TILE_SIZE;

    // Load tiles with halo
    loadTileWithHalo(Sz, s_Sz, r, c, blockStartI, blockStartJ, tx, ty, -9999.0);
    loadTileWithHalo(Sh, s_Sh, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    loadTileWithHalo(ST, s_ST, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    __syncthreads();

    int i = blockStartI + ty;
    int j = blockStartJ + tx;

    if (i >= r || j >= c) return;

    // Shared memory indices (with halo offset)
    int si = ty + HALO_SIZE;
    int sj = tx + HALO_SIZE;

    double h0 = s_Sh[si][sj];
    if (h0 <= 0) return;

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];

    double T = s_ST[si][sj];
    double rr = pow(10.0, _a + _b * T);
    double hc = pow(10.0, _c + _d * T);
    double sz0 = s_Sz[si][sj];

    // Initialize neighbor data - all from shared memory
    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k];
        int nj = j + d_Xj[k];
        int sni = si + d_Xi[k];
        int snj = sj + d_Xj[k];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            eliminated[k] = true;
            continue;
        }

        double sz = s_Sz[sni][snj];
        h[k] = s_Sh[sni][snj];
        w[k] = Pc;
        Pr[k] = rr;

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

    // Minimization algorithm - compute avg outside the loop to match serial version
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

    // Compute outflows - use the final avg computed above
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
// CUDA Kernel: massBalance WITH Halo
// ----------------------------------------------------------------------------
__global__ void kernel_massBalance_halo(
    int r, int c,
    double* Sh, double* Sh_next,
    double* ST, double* ST_next,
    double* Mf)
{
    __shared__ double s_Sh[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_ST[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockStartI = blockIdx.y * TILE_SIZE;
    int blockStartJ = blockIdx.x * TILE_SIZE;

    loadTileWithHalo(Sh, s_Sh, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    loadTileWithHalo(ST, s_ST, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    __syncthreads();

    int i = blockStartI + ty;
    int j = blockStartJ + tx;

    if (i >= r || j >= c) return;

    int si = ty + HALO_SIZE;
    int sj = tx + HALO_SIZE;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};

    double initial_h = s_Sh[si][sj];
    double initial_t = s_ST[si][sj];
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        if (ni < 0 || ni >= r || nj < 0 || nj >= c) continue;

        int sni = si + d_Xi[n];
        int snj = sj + d_Xj[n];

        double neigh_t = s_ST[sni][snj];
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
// CUDA Kernel: computeNewTemperatureAndSolidification WITH Halo
// ----------------------------------------------------------------------------
__global__ void kernel_computeNewTemperatureAndSolidification_halo(
    int r, int c,
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double* Sz, double* Sz_next,
    double* Sh, double* Sh_next,
    double* ST, double* ST_next,
    double* Mhs, bool* Mb)
{
    __shared__ double s_Sz[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_Sh[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_ST[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ bool s_Mb[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockStartI = blockIdx.y * TILE_SIZE;
    int blockStartJ = blockIdx.x * TILE_SIZE;

    loadTileWithHalo(Sz, s_Sz, r, c, blockStartI, blockStartJ, tx, ty, -9999.0);
    loadTileWithHalo(Sh, s_Sh, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    loadTileWithHalo(ST, s_ST, r, c, blockStartI, blockStartJ, tx, ty, 0.0);
    loadTileWithHaloBool(Mb, s_Mb, r, c, blockStartI, blockStartJ, tx, ty, true);
    __syncthreads();

    int i = blockStartI + ty;
    int j = blockStartJ + tx;

    if (i >= r || j >= c) return;

    int si = ty + HALO_SIZE;
    int sj = tx + HALO_SIZE;

    double z = s_Sz[si][sj];
    double h = s_Sh[si][sj];
    double T = s_ST[si][sj];

    if (h > 0 && s_Mb[si][sj] == false) {
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
// Reduction kernel
// ----------------------------------------------------------------------------
__global__ void kernel_reduceAdd(double* input, double* output, int size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double sum = 0.0;
    if (i < size) sum += input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

double reduceAddCUDA(double* d_buffer, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);

    double* d_partial;
    CUDA_CHECK(cudaMallocManaged(&d_partial, numBlocks * sizeof(double)));

    kernel_reduceAdd<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        kernel_reduceAdd<<<newNumBlocks, blockSize, blockSize * sizeof(double)>>>(d_partial, d_partial, numBlocks);
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

    int num_vents = sciara->simulation->vent.size();
    int* d_vent_x;
    int* d_vent_y;
    double* d_vent_thickness;

    CUDA_CHECK(cudaMallocManaged(&d_vent_x, num_vents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_vent_y, num_vents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_vent_thickness, num_vents * sizeof(double)));

    for (int k = 0; k < num_vents; k++) {
        d_vent_x[k] = sciara->simulation->vent[k].x();
        d_vent_y[k] = sciara->simulation->vent[k].y();
    }

    // Block size is TILE_SIZE, not BLOCK_SIZE (which includes halo)
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((c + TILE_SIZE - 1) / TILE_SIZE,
                 (r + TILE_SIZE - 1) / TILE_SIZE);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    while ((max_steps > 0 && sciara->simulation->step < max_steps) &&
           ((sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
            (total_current_lava == -1 || total_current_lava > thickness_threshold)))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        for (int k = 0; k < num_vents; k++) {
            d_vent_thickness[k] = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time,
                sciara->parameters->Pclock,
                sciara->simulation->emission_time,
                sciara->parameters->Pac);
            sciara->simulation->total_emitted_lava += d_vent_thickness[k];
        }

        // 1. Emit Lava
        kernel_emitLava_halo<<<gridDim, blockDim>>>(
            r, c, d_vent_x, d_vent_y, d_vent_thickness, num_vents,
            sciara->parameters->PTvent,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 2. Compute Outflows
        kernel_computeOutflows_halo<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz, sciara->substates->Sh,
            sciara->substates->ST, sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a, sciara->parameters->b,
            sciara->parameters->c, sciara->parameters->d);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Mass Balance
        kernel_massBalance_halo<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 4. Temperature and Solidification
        kernel_computeNewTemperatureAndSolidification_halo<<<gridDim, blockDim>>>(
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

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 5. Reduction
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

    CUDA_CHECK(cudaFree(d_vent_x));
    CUDA_CHECK(cudaFree(d_vent_y));
    CUDA_CHECK(cudaFree(d_vent_thickness));

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}