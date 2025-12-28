/**
 * sciara_fv2_cfame.cu - CUDA CfAMe (Conflict-Free Memory-Equivalent)
 *
 * CfAMe combines computeOutflows and massBalance into a single kernel
 * using atomic operations to handle race conditions when multiple
 * threads update the same cell.
 *
 * ATOMIC SCATTER PATTERN:
 * -----------------------
 * 1. TRADITIONAL vs CfAMe:
 *
 *    Traditional (Gather):
 *    - computeOutflows: Cell writes flows to Mf[8]
 *    - massBalance: Cell reads flows FROM neighbors
 *    - Two kernel launches, Mf intermediate storage
 *
 *    CfAMe (Scatter):
 *    - Single kernel: Cell computes flows AND pushes to neighbors
 *    - atomicAdd(&Sh_next[neighbor], flow)
 *    - No separate massBalance kernel needed
 *
 * 2. WHY ATOMIC OPERATIONS:
 *    - Multiple cells may update same destination simultaneously
 *    - E.g., cell (5,5) receives flows from 8 neighbors
 *    - Without atomics: Race condition, lost updates
 *    - With atomics: Serialized but correct
 *
 * 3. ATOMIC IMPLEMENTATION (FP64):
 *    - CUDA < 6.0: No native atomicAdd for double
 *    - Use atomicCAS loop (compare-and-swap)
 *    - ~100 cycles per atomic vs ~4 cycles for normal write
 *
 * 4. TEMPERATURE ACCUMULATION TRICK:
 *    - Can't atomicAdd weighted average directly
 *    - Instead: ST_next stores h*T (mass-weighted sum)
 *    - After all updates: T = ST_next / Sh_next (normalize)
 *
 * WHY NOT TILED:
 * - Scatter pattern writes to RANDOM neighbor locations
 * - Neighbors may be in different tiles
 * - Shared memory can't help with scattered writes
 * - Global atomics are required regardless
 *
 * MEMORY EQUIVALENT:
 * - Still allocates Mf buffer (for debugging/comparison)
 * - See CfAMo for memory-optimized version without Mf
 *
 * PERFORMANCE: 1.10× speedup vs Global (fewer kernel launches)
 *
 * NUMERICAL NOTE:
 * - Atomic execution order is non-deterministic
 * - May produce slightly different results than Global/Tiled
 * - Differences are within FP64 precision (~1e-14)
 *
 * See REPORT.md Section 2.3 for detailed analysis.
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
// CUDA configuration
// ----------------------------------------------------------------------------
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

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
// Double-precision atomic add (for devices without native support)
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
// CUDA Kernel: emitLava
// ----------------------------------------------------------------------------
__global__ void kernel_emitLava(
    int r, int c,
    int* vent_x, int* vent_y, double* vent_thickness,
    int num_vents,
    double PTvent,
    double* Sh, double* Sh_next,
    double* ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

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
// CUDA Kernel: Initialize buffers for CfAMe
// ----------------------------------------------------------------------------
__global__ void kernel_initBuffers_CfAMe(
    int r, int c,
    double* Sh, double* ST,
    double* Sh_next, double* ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double sh = GET(Sh, c, i, j);
    double st = GET(ST, c, i, j);

    SET(Sh_next, c, i, j, sh);
    SET(ST_next, c, i, j, sh * st);  // Store h*T for accumulation
}

// ----------------------------------------------------------------------------
// CUDA Kernel: CfA_Me - Combined Outflows + Mass Balance
// Each cell computes its outflows and atomically updates neighbors
// ----------------------------------------------------------------------------
__global__ void kernel_CfA_Me(
    int r, int c,
    double* Sz, double* Sh, double* ST,
    double* Sh_next, double* ST_next,
    double* Mf,  // Still keep Mf for intermediate storage
    double Pc, double _a, double _b, double _c, double _d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double h0 = GET(Sh, c, i, j);

    if (h0 <= 0) return;

    // Local arrays for outflow calculation
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

    // Compute outflows and atomically update neighbors - use the final avg computed above
    double total_outflow = 0.0;

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        double flow = 0.0;

        if (!eliminated[k] && h[0] > hc * cos(theta[k])) {
            flow = Pr[k] * (avg - H[k]);
        }

        // Store in Mf for reference
        BUF_SET(Mf, r, c, k - 1, i, j, flow);

        if (flow > 0) {
            int ni = i + d_Xi[k];
            int nj = j + d_Xj[k];

            // Atomically add inflow to neighbor
            atomicAddDouble(&Sh_next[ni * c + nj], flow);
            atomicAddDouble(&ST_next[ni * c + nj], flow * T);

            total_outflow += flow;
        }
    }

    // Subtract total outflow from this cell
    atomicAddDouble(&Sh_next[i * c + j], -total_outflow);
    atomicAddDouble(&ST_next[i * c + j], -total_outflow * T);
}

// ----------------------------------------------------------------------------
// CUDA Kernel: Normalize temperature after CfA_Me
// ----------------------------------------------------------------------------
__global__ void kernel_normalizeTemperature(
    int r, int c,
    double* Sh_next, double* ST_next)
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
    double* Sz, double* Sz_next,
    double* Sh, double* Sh_next,
    double* ST, double* ST_next,
    double* Mhs, bool* Mb)
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

        for (int k = 0; k < num_vents; k++) {
            d_vent_thickness[k] = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time,
                sciara->parameters->Pclock,
                sciara->simulation->emission_time,
                sciara->parameters->Pac);
            sciara->simulation->total_emitted_lava += d_vent_thickness[k];
        }

        // 1. Emit Lava
        kernel_emitLava<<<gridDim, blockDim>>>(
            r, c, d_vent_x, d_vent_y, d_vent_thickness, num_vents,
            sciara->parameters->PTvent,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 2a. Initialize buffers for CfAMe
        kernel_initBuffers_CfAMe<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->ST,
            sciara->substates->Sh_next, sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2b. CfA_Me: Combined Outflows + Mass Balance
        kernel_CfA_Me<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sz,
            sciara->substates->Sh,
            sciara->substates->ST,
            sciara->substates->Sh_next,
            sciara->substates->ST_next,
            sciara->substates->Mf,
            sciara->parameters->Pc,
            sciara->parameters->a, sciara->parameters->b,
            sciara->parameters->c, sciara->parameters->d);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Normalize temperature
        kernel_normalizeTemperature<<<gridDim, blockDim>>>(
            r, c, sciara->substates->Sh_next, sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 3. Temperature and Solidification
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

        CUDA_CHECK(cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

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

    CUDA_CHECK(cudaFree(d_vent_x));
    CUDA_CHECK(cudaFree(d_vent_y));
    CUDA_CHECK(cudaFree(d_vent_thickness));

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}