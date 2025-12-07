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
// Tiling parameters for V4 (CfAMe - Conflict-free Access Memory equivalent)
// ----------------------------------------------------------------------------
#define TILE_SIZE 16
#define PADDED_SIZE 17  // 16 + 1 padding to avoid bank conflicts

// ----------------------------------------------------------------------------
// Read/Write access macros
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// CUDA Kernels - Version 4: CfAMe (Conflict-free Access Memory Equivalent)
// ----------------------------------------------------------------------------

// atomicAdd for double (for SM < 6.0)
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

// Simple kernels kept from V1/V2/V3 (no changes)
__global__ void emitLava_kernel(
    int r, int c,
    int *vent_x, int *vent_y,
    double *vent_thickness_values,
    int num_vents,
    double *Sh, double *Sh_next,
    double *ST_next, double PTvent)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= r || j >= c) return;
    
    for (int k = 0; k < num_vents; k++) {
        if (i == vent_y[k] && j == vent_x[k]) {
            SET(Sh_next, c, i, j, GET(Sh, c, i, j) + vent_thickness_values[k]);
            SET(ST_next, c, i, j, PTvent);
        }
    }
}

// VERSION 4: CfAMe computeOutflows with PADDED shared memory (no bank conflicts)
__global__ void computeOutflows_cfame_kernel(
    int r, int c,
    int *Xi, int *Xj,
    double *Sz, double *Sh, double *ST, double *Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    // PADDED shared memory: [16][17] instead of [16][16] to avoid bank conflicts
    __shared__ double s_Sz[TILE_SIZE][PADDED_SIZE];
    __shared__ double s_Sh[TILE_SIZE][PADDED_SIZE];
    __shared__ double s_ST[TILE_SIZE][PADDED_SIZE];
    
    // Global indices
    int gi = blockIdx.y * TILE_SIZE + threadIdx.y;
    int gj = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Local indices
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    // Load tile into PADDED shared memory
    if (gi < r && gj < c) {
        s_Sz[ti][tj] = GET(Sz, c, gi, gj);
        s_Sh[ti][tj] = GET(Sh, c, gi, gj);
        s_ST[ti][tj] = GET(ST, c, gi, gj);
    } else {
        s_Sz[ti][tj] = 0.0;
        s_Sh[ti][tj] = 0.0;
        s_ST[ti][tj] = 0.0;
    }
    __syncthreads();
    
    // Boundary check
    if (gi >= r || gj >= c) return;
    if (s_Sh[ti][tj] <= 0) return;
    
    // Computation - same as V2 but with conflict-free access
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

    T = s_ST[ti][tj];
    rr = pow(10.0, _a + _b * T);
    hc = pow(10.0, _c + _d * T);

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        sz0 = s_Sz[ti][tj];
        
        // Neighbor indices in shared memory
        int ni = gi + Xi[k];
        int nj = gj + Xj[k];
        int nti = ti + Xi[k];
        int ntj = tj + Xj[k];
        
        // Hybrid access (like V2) but from padded shared memory
        if (nti >= 0 && nti < TILE_SIZE && ntj >= 0 && ntj < TILE_SIZE) {
            // Neighbor in shared memory - CONFLICT-FREE access!
            sz = s_Sz[nti][ntj];
            h[k] = s_Sh[nti][ntj];
        } else {
            // Neighbor outside tile - read from global memory
            if (ni >= 0 && ni < r && nj >= 0 && nj < c) {
                sz = GET(Sz, c, ni, nj);
                h[k] = GET(Sh, c, ni, nj);
            } else {
                sz = sz0;
                h[k] = 0.0;
            }
        }
        
        w[k] = Pc;
        Pr[k] = rr;

        if (k < VON_NEUMANN_NEIGHBORS)
            z[k] = sz;
        else
            z[k] = sz0 - (sz0 - sz) / sqrt(2.0);
    }

    // Rest of computation (same as V1/V2/V3)
    H[0] = z[0];
    theta[0] = 0;
    eliminated[0] = false;
    
    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
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
        for (int k = 0; k < MOORE_NEIGHBORS; k++) {
            if (!eliminated[k]) {
                avg += H[k];
                counter++;
            }
        }
        if (counter != 0)
            avg = avg / double(counter);
        for (int k = 0; k < MOORE_NEIGHBORS; k++) {
            if (!eliminated[k] && avg <= H[k]) {
                eliminated[k] = true;
                loop = true;
            }
        }
    } while (loop);

    for (int k = 1; k < MOORE_NEIGHBORS; k++) {
        if (!eliminated[k] && h[0] > hc * cos(theta[k]))
            BUF_SET(Mf, r, c, k - 1, gi, gj, Pr[k] * (avg - H[k]));
        else
            BUF_SET(Mf, r, c, k - 1, gi, gj, 0.0);
    }
}

// VERSION 4: CfAMe massBalance with PADDED shared memory
__global__ void massBalance_cfame_kernel(
    int r, int c,
    int *Xi, int *Xj,
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mf)
{
    // PADDED shared memory for conflict-free access
    __shared__ double s_ST[TILE_SIZE][PADDED_SIZE];
    
    int gi = blockIdx.y * TILE_SIZE + threadIdx.y;
    int gj = blockIdx.x * TILE_SIZE + threadIdx.x;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    // Load ST into padded shared memory
    if (gi < r && gj < c) {
        s_ST[ti][tj] = GET(ST, c, gi, gj);
    } else {
        s_ST[ti][tj] = 0.0;
    }
    __syncthreads();
    
    if (gi >= r || gj >= c) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
    double inFlow, outFlow, neigh_t;
    double initial_h = GET(Sh, c, gi, gj);
    double initial_t = s_ST[ti][tj];
    double h_next = initial_h;
    double t_next = initial_h * initial_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = gi + Xi[n];
        int nj = gj + Xj[n];
        int nti = ti + Xi[n];
        int ntj = tj + Xj[n];
        
        // Hybrid access with conflict-free shared memory
        if (nti >= 0 && nti < TILE_SIZE && ntj >= 0 && ntj < TILE_SIZE) {
            neigh_t = s_ST[nti][ntj];  // Conflict-free!
        } else {
            neigh_t = GET(ST, c, ni, nj);
        }
        
        inFlow = BUF_GET(Mf, r, c, inflowsIndices[n - 1], ni, nj);
        outFlow = BUF_GET(Mf, r, c, n - 1, gi, gj);

        h_next += inFlow - outFlow;
        t_next += (inFlow * neigh_t - outFlow * initial_t);
    }

    if (h_next > 0) {
        t_next /= h_next;
        SET(ST_next, c, gi, gj, t_next);
        SET(Sh_next, c, gi, gj, h_next);
    }
}

// Keep V1/V2/V3 versions unchanged
__global__ void computeNewTemperatureAndSolidification_kernel(
    int r, int c,
    double Pepsilon, double Psigma, double Pclock, double Pcool,
    double Prho, double Pcv, double Pac, double PTsol,
    double *Sz, double *Sz_next,
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mhs, bool *Mb)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
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

__global__ void boundaryConditions_kernel(
    int r, int c,
    bool *Mb,
    double *Sh_next,
    double *ST_next)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= r || j >= c) return;
    
    if (GET(Mb, c, i, j)) {
        SET(Sh_next, c, i, j, 0.0);
        SET(ST_next, c, i, j, 0.0);
    }
}

// Parallel reduction kernel for sum (improved version)
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

// ----------------------------------------------------------------------------
// Host wrapper functions
// ----------------------------------------------------------------------------

void emitLava_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    int num_vents = sciara->simulation->vent.size();
    if (num_vents == 0) return;
    
    int *d_vent_x, *d_vent_y;
    double *d_vent_thickness;
    
    cudaMalloc(&d_vent_x, sizeof(int) * num_vents);
    cudaMalloc(&d_vent_y, sizeof(int) * num_vents);
    cudaMalloc(&d_vent_thickness, sizeof(double) * num_vents);
    
    int *h_vent_x = new int[num_vents];
    int *h_vent_y = new int[num_vents];
    double *h_vent_thickness = new double[num_vents];
    
    for (int k = 0; k < num_vents; k++) {
        h_vent_x[k] = sciara->simulation->vent[k].x();
        h_vent_y[k] = sciara->simulation->vent[k].y();
        h_vent_thickness[k] = sciara->simulation->vent[k].thickness(
            sciara->simulation->elapsed_time,
            sciara->parameters->Pclock,
            sciara->simulation->emission_time,
            sciara->parameters->Pac);
        // Track total emitted lava
        sciara->simulation->total_emitted_lava += h_vent_thickness[k];
    }
    
    cudaMemcpy(d_vent_x, h_vent_x, sizeof(int) * num_vents, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vent_y, h_vent_y, sizeof(int) * num_vents, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vent_thickness, h_vent_thickness, sizeof(double) * num_vents, cudaMemcpyHostToDevice);
    
    emitLava_kernel<<<grid, block>>>(
        sciara->domain->rows,
        sciara->domain->cols,
        d_vent_x, d_vent_y,
        d_vent_thickness,
        num_vents,
        sciara->substates->Sh,
        sciara->substates->Sh_next,
        sciara->substates->ST_next,
        sciara->parameters->PTvent);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_vent_x);
    cudaFree(d_vent_y);
    cudaFree(d_vent_thickness);
    delete[] h_vent_x;
    delete[] h_vent_y;
    delete[] h_vent_thickness;
}

void computeOutflows_cfame_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    int *d_Xi, *d_Xj;
    cudaMalloc(&d_Xi, sizeof(int) * MOORE_NEIGHBORS);
    cudaMalloc(&d_Xj, sizeof(int) * MOORE_NEIGHBORS);
    cudaMemcpy(d_Xi, sciara->X->Xi, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    
    computeOutflows_cfame_kernel<<<grid, block>>>(
        sciara->domain->rows,
        sciara->domain->cols,
        d_Xi, d_Xj,
        sciara->substates->Sz,
        sciara->substates->Sh,
        sciara->substates->ST,
        sciara->substates->Mf,
        sciara->parameters->Pc,
        sciara->parameters->a,
        sciara->parameters->b,
        sciara->parameters->c,
        sciara->parameters->d);
    
    cudaDeviceSynchronize();
    cudaFree(d_Xi);
    cudaFree(d_Xj);
}

void massBalance_cfame_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    int *d_Xi, *d_Xj;
    cudaMalloc(&d_Xi, sizeof(int) * MOORE_NEIGHBORS);
    cudaMalloc(&d_Xj, sizeof(int) * MOORE_NEIGHBORS);
    cudaMemcpy(d_Xi, sciara->X->Xi, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    
    massBalance_cfame_kernel<<<grid, block>>>(
        sciara->domain->rows,
        sciara->domain->cols,
        d_Xi, d_Xj,
        sciara->substates->Sh,
        sciara->substates->Sh_next,
        sciara->substates->ST,
        sciara->substates->ST_next,
        sciara->substates->Mf);
    
    cudaDeviceSynchronize();
    cudaFree(d_Xi);
    cudaFree(d_Xj);
}

void computeNewTemperatureAndSolidification_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    computeNewTemperatureAndSolidification_kernel<<<grid, block>>>(
        sciara->domain->rows,
        sciara->domain->cols,
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
    
    cudaDeviceSynchronize();
}

void boundaryConditions_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    boundaryConditions_kernel<<<grid, block>>>(
        sciara->domain->rows,
        sciara->domain->cols,
        sciara->substates->Mb,
        sciara->substates->Sh_next,
        sciara->substates->ST_next);
    
    cudaDeviceSynchronize();
}

double reduceAdd_cuda(int r, int c, double *buffer)
{
    int size = r * c;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    double* d_partial;
    cudaMallocManaged(&d_partial, blocksPerGrid * sizeof(double));

    // First reduction pass
    reduceAddKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(buffer, d_partial, size);
    cudaDeviceSynchronize();

    // Continue reduction until we have a single value
    while (blocksPerGrid > 1) {
        int n = blocksPerGrid;
        blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        reduceAddKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_partial, d_partial, n);
        cudaDeviceSynchronize();
    }

    double result = d_partial[0];
    cudaFree(d_partial);

    return result;
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Sciara *sciara;
    init(sciara);

    int max_steps = atoi(argv[MAX_STEPS_ID]);
    loadConfiguration(argv[INPUT_PATH_ID], sciara);

    // Setup CUDA grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads per block
    dim3 grid((sciara->domain->cols + block.x - 1) / block.x,
              (sciara->domain->rows + block.y - 1) / block.y);

    double total_current_lava = -1;
    simulationInitialize(sciara);

    util::Timer cl_timer;

    int reduceInterval = atoi(argv[REDUCE_INTERVL_ID]);
    double thickness_threshold = atof(argv[THICKNESS_THRESHOLD_ID]);
    
    while ((max_steps > 0 && sciara->simulation->step < max_steps) || 
           (sciara->simulation->elapsed_time <= sciara->simulation->effusion_duration) || 
           (total_current_lava == -1 || total_current_lava > thickness_threshold))
    {
        sciara->simulation->elapsed_time += sciara->parameters->Pclock;
        sciara->simulation->step++;

        // Apply the emitLava kernel
        emitLava_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        // Apply the CfAMe computeOutflows kernel
        computeOutflows_cfame_cuda(sciara, grid, block);

        // Apply the CfAMe massBalance kernel
        massBalance_cfame_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        // Apply the computeNewTemperatureAndSolidification kernel
        computeNewTemperatureAndSolidification_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sz, sciara->substates->Sz_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        // Apply the boundaryConditions kernel
        boundaryConditions_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        // Global reduction
        if (sciara->simulation->step % reduceInterval == 0)
            total_current_lava = reduceAdd_cuda(sciara->domain->rows, sciara->domain->cols, 
                                                sciara->substates->Sh);
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    // CRITICAL FIX for bus error: Copy to CPU buffers before save
    cudaDeviceSynchronize();
    
    int mem_size = sciara->domain->rows * sciara->domain->cols;
    
    double *Sz_host = (double*)malloc(sizeof(double) * mem_size);
    double *Sh_host = (double*)malloc(sizeof(double) * mem_size);
    double *ST_host = (double*)malloc(sizeof(double) * mem_size);
    double *Mhs_host = (double*)malloc(sizeof(double) * mem_size);
    
    cudaMemcpy(Sz_host, sciara->substates->Sz, sizeof(double) * mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sh_host, sciara->substates->Sh, sizeof(double) * mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ST_host, sciara->substates->ST, sizeof(double) * mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Mhs_host, sciara->substates->Mhs, sizeof(double) * mem_size, cudaMemcpyDeviceToHost);
    
    double *Sz_unified = sciara->substates->Sz;
    double *Sh_unified = sciara->substates->Sh;
    double *ST_unified = sciara->substates->ST;
    double *Mhs_unified = sciara->substates->Mhs;
    
    sciara->substates->Sz = Sz_host;
    sciara->substates->Sh = Sh_host;
    sciara->substates->ST = ST_host;
    sciara->substates->Mhs = Mhs_host;
    
    printf("Saving output to %s...\n", argv[OUTPUT_PATH_ID]);
    saveConfiguration(argv[OUTPUT_PATH_ID], sciara);
    
    sciara->substates->Sz = Sz_unified;
    sciara->substates->Sh = Sh_unified;
    sciara->substates->ST = ST_unified;
    sciara->substates->Mhs = Mhs_unified;
    
    free(Sz_host);
    free(Sh_host);
    free(ST_host);
    free(Mhs_host);

    printf("Releasing memory...\n");
    finalize(sciara);

    return 0;
}
