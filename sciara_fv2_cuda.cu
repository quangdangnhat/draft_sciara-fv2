#include "Sciara.h"
#include "io.h"
#include "util.hpp"
#include <cuda_runtime.h>

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

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multi layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// Device constants for neighborhood
// ----------------------------------------------------------------------------
__constant__ int d_Xi[MOORE_NEIGHBORS];
__constant__ int d_Xj[MOORE_NEIGHBORS];

// ----------------------------------------------------------------------------
// CUDA error checking macro
// ----------------------------------------------------------------------------
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
// CUDA Kernels - Global Memory Version
// ----------------------------------------------------------------------------

__global__ void emitLavaKernel(
    int r,
    int c,
    int numVents,
    int* ventX,
    int* ventY,
    double* ventThickness,
    double PTvent,
    double *Sh,
    double *Sh_next,
    double *ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    for (int k = 0; k < numVents; k++) {
        if (i == ventY[k] && j == ventX[k]) {
            SET(Sh_next, c, i, j, GET(Sh, c, i, j) + ventThickness[k]);
            SET(ST_next, c, i, j, PTvent);
        }
    }
}

__global__ void computeOutflowsKernel(
    int r,
    int c,
    double *Sz,
    double *Sh,
    double *ST,
    double *Mf,
    double Pc,
    double _a,
    double _b,
    double _c,
    double _d)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    if (GET(Sh, c, i, j) <= 0) {
        // Clear outflows for cells without lava
        for (int k = 0; k < NUMBER_OF_OUTFLOWS; k++)
            BUF_SET(Mf, r, c, k, i, j, 0.0);
        return;
    }

    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];
    bool loop;
    int counter;
    double sz0, sz, T, avg, rr, hc;

    T = GET(ST, c, i, j);
    rr = pow(10.0, _a + _b * T);
    hc = pow(10.0, _c + _d * T);

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = i + d_Xi[k];
        int nj = j + d_Xj[k];

        // Boundary check - eliminate neighbors outside domain
        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            eliminated[k] = true;
            z[k] = 0;
            h[k] = 0;
            w[k] = Pc;
            Pr[k] = rr;
            continue;
        }

        sz0 = GET(Sz, c, i, j);
        sz = GET(Sz, c, ni, nj);
        h[k] = GET(Sh, c, ni, nj);
        w[k] = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);

        eliminated[k] = false;  // Initialize as not eliminated
    }

    H[0] = z[0];
    theta[0] = 0;
    eliminated[0] = false;
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (eliminated[k]) continue;  // Skip already eliminated neighbors

        if (z[0] + h[0] > z[k] + h[k]) {
            H[k] = z[k] + h[k];
            theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);
            eliminated[k] = false;
        } else {
            eliminated[k] = true;
        }
    }

    do {
        loop = false;
        avg = h[0];
        counter = 0;
        for (int k = 0; k < MOORE_NEIGHBORS; k++)
            if (!eliminated[k]) {
                avg += H[k];
                counter++;
            }
        if (counter != 0)
            avg = avg / (double)(counter);
        for (int k = 0; k < MOORE_NEIGHBORS; k++)
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k] = true;
                loop = true;
            }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++)
        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
            BUF_SET(Mf, r, c, k - 1, i, j, Pr[k] * (avg - H[k]));
        else
            BUF_SET(Mf, r, c, k - 1, i, j, 0.0);
}

__global__ void massBalanceKernel(
    int r,
    int c,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mf)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
    double inFlow;
    double outFlow;
    double neigh_t;
    double initial_h = GET(Sh, c, i, j);
    double initial_t = GET(ST, c, i, j);
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = i + d_Xi[n];
        int nj = j + d_Xj[n];

        // Boundary check
        if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
            outFlow = BUF_GET(Mf, r, c, n - 1, i, j);
            h_next -= outFlow;
            t_next -= outFlow * initial_t;
            continue;
        }

        neigh_t = GET(ST, c, ni, nj);
        inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], ni, nj);
        outFlow = BUF_GET(Mf, r, c, n - 1, i, j);

        h_next += inFlow - outFlow;
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) {
        t_next /= h_next;
        SET(ST_next, c, i, j, t_next);
        SET(Sh_next, c, i, j, h_next);
    }
}

__global__ void computeNewTemperatureAndSolidificationKernel(
    int r,
    int c,
    double Pepsilon,
    double Psigma,
    double Pclock,
    double Pcool,
    double Prho,
    double Pcv,
    double Pac,
    double PTsol,
    double *Sz,
    double *Sz_next,
    double *Sh,
    double *Sh_next,
    double *ST,
    double *ST_next,
    double *Mhs,
    bool *Mb)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    double nT, aus;
    double z = GET(Sz, c, i, j);
    double h = GET(Sh, c, i, j);
    double T = GET(ST, c, i, j);

    if (h > 0 && GET(Mb, c, i, j) == false) {
        aus = 1.0 + (3 * pow(T, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * Pac);
        nT = T / pow(aus, 1.0 / 3.0);

        if (nT > PTsol) // no solidification
            SET(ST_next, c, i, j, nT);
        else { // solidification
            SET(Sz_next, c, i, j, z + h);
            SET(Sh_next, c, i, j, 0.0);
            SET(ST_next, c, i, j, PTsol);
            SET(Mhs, c, i, j, GET(Mhs, c, i, j) + h);
        }
    }
}

__global__ void boundaryConditionsKernel(
    int r,
    int c,
    bool *Mb,
    double *Sh_next,
    double *ST_next)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= r || j >= c) return;

    // Note: Original implementation returns early, keeping same behavior
    return;

    if (GET(Mb, c, i, j)) {
        SET(Sh_next, c, i, j, 0.0);
        SET(ST_next, c, i, j, 0.0);
    }
}

__global__ void copyBufferKernel(double* dst, double* src, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// Parallel reduction kernel for sum
__global__ void reduceAddKernel(double* input, double* output, int n)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and add two elements per thread
    double mySum = 0;
    if (i < n) mySum = input[i];
    if (i + blockDim.x < n) mySum += input[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

double reduceAddGPU(double* d_buffer, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    double* d_partial;
    double* d_result;
    CUDA_CHECK(cudaMallocManaged(&d_partial, blocksPerGrid * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&d_result, sizeof(double)));

    // First reduction pass
    reduceAddKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_buffer, d_partial, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Continue reduction until we have a single value
    while (blocksPerGrid > 1) {
        int n = blocksPerGrid;
        blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        reduceAddKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_partial, d_partial, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double result = d_partial[0];

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

// ----------------------------------------------------------------------------
// CUDA Memory allocation functions
// ----------------------------------------------------------------------------
void allocateSubstatesCUDA(Sciara *sciara)
{
    int size = sciara->domain->rows * sciara->domain->cols;

    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mf, size * NUMBER_OF_OUTFLOWS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mb, size * sizeof(bool)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mhs, size * sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(sciara->substates->Sz, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->Sz_next, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->Sh, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->Sh_next, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->ST, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->ST_next, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->Mf, 0, size * NUMBER_OF_OUTFLOWS * sizeof(double)));
    CUDA_CHECK(cudaMemset(sciara->substates->Mb, 0, size * sizeof(bool)));
    CUDA_CHECK(cudaMemset(sciara->substates->Mhs, 0, size * sizeof(double)));
}

void deallocateSubstatesCUDA(Sciara *sciara)
{
    if(sciara->substates->Sz) cudaFree(sciara->substates->Sz);
    if(sciara->substates->Sz_next) cudaFree(sciara->substates->Sz_next);
    if(sciara->substates->Sh) cudaFree(sciara->substates->Sh);
    if(sciara->substates->Sh_next) cudaFree(sciara->substates->Sh_next);
    if(sciara->substates->ST) cudaFree(sciara->substates->ST);
    if(sciara->substates->ST_next) cudaFree(sciara->substates->ST_next);
    if(sciara->substates->Mf) cudaFree(sciara->substates->Mf);
    if(sciara->substates->Mb) cudaFree(sciara->substates->Mb);
    if(sciara->substates->Mhs) cudaFree(sciara->substates->Mhs);
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 6) {
        printf("Usage: %s <input_config> <output_path> <max_steps> <reduce_interval> <thickness_threshold>\n", argv[0]);
        return 1;
    }

    // Initialize CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    Sciara *sciara;
    init(sciara);

    // Input data
    int max_steps = atoi(argv[MAX_STEPS_ID]);

    // Temporarily use CPU allocation for loading
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    // Copy data to managed memory
    int rows = sciara->domain->rows;
    int cols = sciara->domain->cols;
    int size = rows * cols;

    printf("Domain size: %d x %d = %d cells\n", rows, cols, size);

    // Store original pointers
    double* h_Sz = sciara->substates->Sz;
    double* h_Sz_next = sciara->substates->Sz_next;
    double* h_Sh = sciara->substates->Sh;
    double* h_Sh_next = sciara->substates->Sh_next;
    double* h_ST = sciara->substates->ST;
    double* h_ST_next = sciara->substates->ST_next;
    double* h_Mf = sciara->substates->Mf;
    bool* h_Mb = sciara->substates->Mb;
    double* h_Mhs = sciara->substates->Mhs;

    // Allocate CUDA managed memory
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST_next, size * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mf, size * NUMBER_OF_OUTFLOWS * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mb, size * sizeof(bool)));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mhs, size * sizeof(double)));

    // Copy data from host to managed memory
    memcpy(sciara->substates->Sz, h_Sz, size * sizeof(double));
    memcpy(sciara->substates->Sz_next, h_Sz_next, size * sizeof(double));
    memcpy(sciara->substates->Sh, h_Sh, size * sizeof(double));
    memcpy(sciara->substates->Sh_next, h_Sh_next, size * sizeof(double));
    memcpy(sciara->substates->ST, h_ST, size * sizeof(double));
    memcpy(sciara->substates->ST_next, h_ST_next, size * sizeof(double));
    memcpy(sciara->substates->Mf, h_Mf, size * NUMBER_OF_OUTFLOWS * sizeof(double));
    memcpy(sciara->substates->Mb, h_Mb, size * sizeof(bool));
    memcpy(sciara->substates->Mhs, h_Mhs, size * sizeof(double));

    // Free original host memory
    delete[] h_Sz;
    delete[] h_Sz_next;
    delete[] h_Sh;
    delete[] h_Sh_next;
    delete[] h_ST;
    delete[] h_ST_next;
    delete[] h_Mf;
    delete[] h_Mb;
    delete[] h_Mhs;

    // Copy neighborhood to constant memory
    int Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
    int Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xi, Xi, MOORE_NEIGHBORS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Xj, Xj, MOORE_NEIGHBORS * sizeof(int)));

    // Allocate vent data on GPU
    int numVents = sciara->simulation->vent.size();
    int* d_ventX;
    int* d_ventY;
    double* d_ventThickness;

    CUDA_CHECK(cudaMallocManaged(&d_ventX, numVents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_ventY, numVents * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&d_ventThickness, numVents * sizeof(double)));

    for (int k = 0; k < numVents; k++) {
        d_ventX[k] = sciara->simulation->vent[k].x();
        d_ventY[k] = sciara->simulation->vent[k].y();
    }

    // Simulation initialization
    simulationInitialize(sciara);

    // CUDA grid configuration
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);

    int copyBlockSize = 256;
    int copyGridSize = (size + copyBlockSize - 1) / copyBlockSize;

    printf("Grid size: %d x %d blocks\n", gridSize.x, gridSize.y);
    printf("Block size: %d x %d threads\n", blockSize.x, blockSize.y);

    // Simulation loop
    double total_current_lava = -1;
    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);

    util::Timer cl_timer;

    while ((max_steps > 0 && sciara->simulation->step < max_steps) ||
           (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) ||
           (total_current_lava == -1 || total_current_lava > thickness_threshold))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // Update vent thickness for current time step
        for (int k = 0; k < numVents; k++) {
            d_ventThickness[k] = sciara->simulation->vent[k].thickness(
                sciara->simulation->elapsed_time,
                sciara->parameters->Pclock,
                sciara->simulation->emission_time,
                sciara->parameters->Pac);
            sciara->simulation->total_emitted_lava += d_ventThickness[k];
        }

        // emitLava kernel
        emitLavaKernel<<<gridSize, blockSize>>>(
            rows, cols, numVents,
            d_ventX, d_ventY, d_ventThickness,
            sciara->parameters->PTvent,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy Sh_next -> Sh, ST_next -> ST
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->Sh, sciara->substates->Sh_next, size);
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->ST, sciara->substates->ST_next, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // computeOutflows kernel
        computeOutflowsKernel<<<gridSize, blockSize>>>(
            rows, cols,
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

        // massBalance kernel
        massBalanceKernel<<<gridSize, blockSize>>>(
            rows, cols,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mf);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy Sh_next -> Sh, ST_next -> ST
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->Sh, sciara->substates->Sh_next, size);
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->ST, sciara->substates->ST_next, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // computeNewTemperatureAndSolidification kernel
        computeNewTemperatureAndSolidificationKernel<<<gridSize, blockSize>>>(
            rows, cols,
            sciara->parameters->Pepsilon,
            sciara->parameters->Psigma,
            sciara->parameters->Pclock,
            sciara->parameters->Pcool,
            sciara->parameters->Prho,
            sciara->parameters->Pcv,
            sciara->parameters->Pac,
            sciara->parameters->PTsol,
            sciara->substates->Sz,
            sciara->substates->Sz_next,
            sciara->substates->Sh,
            sciara->substates->Sh_next,
            sciara->substates->ST,
            sciara->substates->ST_next,
            sciara->substates->Mhs,
            sciara->substates->Mb);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy all state variables
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->Sz, sciara->substates->Sz_next, size);
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->Sh, sciara->substates->Sh_next, size);
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->ST, sciara->substates->ST_next, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // boundaryConditions kernel
        boundaryConditionsKernel<<<gridSize, blockSize>>>(
            rows, cols,
            sciara->substates->Mb,
            sciara->substates->Sh_next,
            sciara->substates->ST_next);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy Sh_next -> Sh, ST_next -> ST
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->Sh, sciara->substates->Sh_next, size);
        copyBufferKernel<<<copyGridSize, copyBlockSize>>>(sciara->substates->ST, sciara->substates->ST_next, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Global reduction
        if (sciara->simulation->step % reduceInterval == 0) {
            total_current_lava = reduceAddGPU(sciara->substates->Sh, size);
        }
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);

    // Synchronize before saving (ensure all GPU operations complete)
    CUDA_CHECK(cudaDeviceSynchronize());
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);

    printf("Releasing memory...\n");

    // Free vent data
    CUDA_CHECK(cudaFree(d_ventX));
    CUDA_CHECK(cudaFree(d_ventY));
    CUDA_CHECK(cudaFree(d_ventThickness));

    // Free substates (using CUDA free)
    deallocateSubstatesCUDA(sciara);

    // Set to NULL to prevent double free in finalize
    sciara->substates->Sz = NULL;
    sciara->substates->Sz_next = NULL;
    sciara->substates->Sh = NULL;
    sciara->substates->Sh_next = NULL;
    sciara->substates->ST = NULL;
    sciara->substates->ST_next = NULL;
    sciara->substates->Mf = NULL;
    sciara->substates->Mb = NULL;
    sciara->substates->Mhs = NULL;

    // Finalize remaining structures
    delete sciara->domain;
    delete[] sciara->X->Xi;
    delete[] sciara->X->Xj;
    delete sciara->X;
    delete sciara->substates;
    delete sciara->parameters;
    delete sciara->simulation;
    delete sciara;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
