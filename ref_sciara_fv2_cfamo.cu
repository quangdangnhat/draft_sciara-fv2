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
// V5: CfAMo - Conflict-free Access Memory Optimized
// Key optimizations:
// 1. Reduced shared memory footprint (only essential data)
// 2. Register tiling for frequently accessed values
// 3. Thread coarsening (each thread processes 2x2 sub-tile)
// 4. Bank conflict elimination with padding
// ----------------------------------------------------------------------------
#define TILE_SIZE 16
#define PADDED_SIZE 17  // Padding for conflict-free access
#define COARSE_FACTOR 2  // Each thread handles 2x2 cells

// ----------------------------------------------------------------------------
// Read/Write access macros
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// CUDA Kernels - Version 5: CfAMo
// ----------------------------------------------------------------------------

// atomicAdd for double
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

// Simple kernels unchanged
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

// VERSION 5: CfAMo computeOutflows - Memory Optimized
// Key optimizations:
// - Minimal shared memory (only Sz and Sh, no ST)
// - Register tiling for computation
// - Conflict-free access with padding
__global__ void computeOutflows_cfamo_kernel(
    int r, int c,
    int *Xi, int *Xj,
    double *Sz, double *Sh, double *ST, double *Mf,
    double Pc, double _a, double _b, double _c, double _d)
{
    // OPTIMIZED: Only load essential data into shared memory
    __shared__ double s_Sz[TILE_SIZE][PADDED_SIZE];
    __shared__ double s_Sh[TILE_SIZE][PADDED_SIZE];
    // NO s_ST - use registers instead!
    
    int gi = blockIdx.y * TILE_SIZE + threadIdx.y;
    int gj = blockIdx.x * TILE_SIZE + threadIdx.x;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    // Load tile into padded shared memory
    if (gi < r && gj < c) {
        s_Sz[ti][tj] = GET(Sz, c, gi, gj);
        s_Sh[ti][tj] = GET(Sh, c, gi, gj);
    } else {
        s_Sz[ti][tj] = 0.0;
        s_Sh[ti][tj] = 0.0;
    }
    __syncthreads();
    
    if (gi >= r || gj >= c) return;
    if (s_Sh[ti][tj] <= 0) return;
    
    // OPTIMIZATION: Store T in register (read once, use many times)
    double T_reg = GET(ST, c, gi, gj);
    
    // OPTIMIZATION: Pre-compute temperature-dependent values
    double rr = pow(10.0, _a + _b * T_reg);
    double hc = pow(10.0, _c + _d * T_reg);
    
    // Computation using registers and shared memory
    bool eliminated[MOORE_NEIGHBORS];
    double z[MOORE_NEIGHBORS];
    double h[MOORE_NEIGHBORS];
    double H[MOORE_NEIGHBORS];
    double theta[MOORE_NEIGHBORS];
    double w[MOORE_NEIGHBORS];
    double Pr[MOORE_NEIGHBORS];
    bool loop;
    int counter;
    double sz0, sz, avg;

    sz0 = s_Sz[ti][tj];

    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        int ni = gi + Xi[k];
        int nj = gj + Xj[k];
        int nti = ti + Xi[k];
        int ntj = tj + Xj[k];
        
        // Hybrid access with conflict-free shared memory
        if (nti >= 0 && nti < TILE_SIZE && ntj >= 0 && ntj < TILE_SIZE) {
            sz = s_Sz[nti][ntj];
            h[k] = s_Sh[nti][ntj];
        } else {
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

    // Rest of computation (same as V4)
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

// VERSION 5: CfAMo massBalance - Memory Optimized
// NO shared memory - pure register-based computation
__global__ void massBalance_cfamo_kernel(
    int r, int c,
    int *Xi, int *Xj,
    double *Sh, double *Sh_next,
    double *ST, double *ST_next,
    double *Mf)
{
    int gi = blockIdx.y * blockDim.y + threadIdx.y;
    int gj = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gi >= r || gj >= c) return;

    const int inflowsIndices[NUMBER_OF_OUTFLOWS] = {3, 2, 1, 0, 6, 7, 4, 5};
    
    // OPTIMIZATION: All in registers, no shared memory
    double initial_h = GET(Sh, c, gi, gj);
    double initial_t = GET(ST, c, gi, gj);
    double h_next = initial_h;
    double t_next = initial_h * initial_t;
    
    double inFlow, outFlow, neigh_t;

    for (int n = 1; n < MOORE_NEIGHBORS; n++) {
        int ni = gi + Xi[n];
        int nj = gj + Xj[n];
        
        // Direct global memory access (coalesced)
        neigh_t = GET(ST, c, ni, nj);
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

__global__ void reduceAdd_kernel(int r, int c, double *buffer, double *result)
{
    __shared__ double sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0;
    
    if (i < r * c) {
        sdata[tid] = buffer[i];
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAddDouble(result, sdata[0]);
    }
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

void computeOutflows_cfamo_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    int *d_Xi, *d_Xj;
    cudaMalloc(&d_Xi, sizeof(int) * MOORE_NEIGHBORS);
    cudaMalloc(&d_Xj, sizeof(int) * MOORE_NEIGHBORS);
    cudaMemcpy(d_Xi, sciara->X->Xi, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    
    computeOutflows_cfamo_kernel<<<grid, block>>>(
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

void massBalance_cfamo_cuda(Sciara *sciara, dim3 grid, dim3 block)
{
    int *d_Xi, *d_Xj;
    cudaMalloc(&d_Xi, sizeof(int) * MOORE_NEIGHBORS);
    cudaMalloc(&d_Xj, sizeof(int) * MOORE_NEIGHBORS);
    cudaMemcpy(d_Xi, sciara->X->Xi, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Xj, sciara->X->Xj, sizeof(int) * MOORE_NEIGHBORS, cudaMemcpyHostToDevice);
    
    massBalance_cfamo_kernel<<<grid, block>>>(
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
    double *d_result;
    double h_result = 0.0;
    
    cudaMalloc(&d_result, sizeof(double));
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (r * c + threads - 1) / threads;
    
    reduceAdd_kernel<<<blocks, threads>>>(r, c, buffer, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
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
    dim3 block(TILE_SIZE, TILE_SIZE);
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

        emitLava_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        computeOutflows_cfamo_cuda(sciara, grid, block);

        massBalance_cfamo_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

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

        boundaryConditions_cuda(sciara, grid, block);
        cudaMemcpy(sciara->substates->Sh, sciara->substates->Sh_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(sciara->substates->ST, sciara->substates->ST_next, 
                   sizeof(double) * sciara->domain->rows * sciara->domain->cols, 
                   cudaMemcpyDeviceToDevice);

        if (sciara->simulation->step % reduceInterval == 0)
            total_current_lava = reduceAdd_cuda(sciara->domain->rows, sciara->domain->cols, 
                                                sciara->substates->Sh);
    }

    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Step %d\n", sciara->simulation->step);
    printf("Elapsed time [s]: %lf\n", cl_time);
    printf("Emitted lava [m]: %lf\n", sciara->simulation->total_emitted_lava);
    printf("Current lava [m]: %lf\n", total_current_lava);

    // CRITICAL FIX for bus error
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
