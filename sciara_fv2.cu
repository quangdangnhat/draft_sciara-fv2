/**
 * sciara_fv2.cu - CUDA Global Memory Version
 *
 * CUDA implementation using only global memory.
 * All kernels access data directly from global memory.
 *
 * Compile: nvcc -O3 sciara_fv2.cu Sciara.cu [other files] -o sciara_cuda
 */

#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
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

    // Check if this cell is a vent
    for (int k = 0; k < num_vents; k++) {
        if (i == vent_y[k] && j == vent_x[k]) {
            double current_h = GET(Sh, c, i, j);
            SET(Sh_next, c, i, j, current_h + vent_thickness[k]);
            SET(ST_next, c, i, j, PTvent);
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernel: computeOutflows
// ----------------------------------------------------------------------------
__global__ void kernel_computeOutflows(
    int r, int c,
    double* Sz, double* Sh, double* ST, double* Mf,
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

        if (counter != 0)
            avg = avg / (double)counter;

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
// CUDA Kernel: massBalance
// ----------------------------------------------------------------------------
__global__ void kernel_massBalance(
    int r, int c,
    double* Sh, double* Sh_next,
    double* ST, double* ST_next,
    double* Mf)
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
// CUDA Kernel: Reduction (for total lava calculation)
// ----------------------------------------------------------------------------
__global__ void kernel_reduceAdd(double* input, double* output, int size)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and do first reduction during load
    double sum = 0.0;
    if (i < size) sum += input[i];
    if (i + blockDim.x < size) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Host function for reduction
double reduceAddCUDA(double* d_buffer, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);

    double* d_partial;
    double* d_result;
    CUDA_CHECK(cudaMallocManaged(&d_partial, numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&d_result, sizeof(double)));

    // First reduction pass
    kernel_reduceAdd<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Continue reducing until single value
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        kernel_reduceAdd<<<newNumBlocks, blockSize, blockSize * sizeof(double)>>>(d_partial, d_partial, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        numBlocks = newNumBlocks;
    }

    // Copy result back
    double result = d_partial[0];

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));

    return result;
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

    // Prepare vent data for GPU
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

    // CUDA grid/block configuration
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

        // Update vent thickness for current time step
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
            r, c,
            d_vent_x, d_vent_y, d_vent_thickness,
            num_vents,
            sciara->parameters->PTvent,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy next to current
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 2. Compute Outflows
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
        CUDA_CHECK(cudaDeviceSynchronize());

        // 3. Mass Balance
        kernel_massBalance<<<gridDim, blockDim>>>(
            r, c,
            sciara->substates->Sh, sciara->substates->Sh_next,
            sciara->substates->ST, sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy next to current
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 4. Temperature and Solidification
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

        // Copy next to current
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next,
                   sizeof(double) * r * c, cudaMemcpyDeviceToDevice));

        // 5. Global reduction (periodically)
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
    CUDA_CHECK(cudaFree(d_vent_x));
    CUDA_CHECK(cudaFree(d_vent_y));
    CUDA_CHECK(cudaFree(d_vent_thickness));

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
