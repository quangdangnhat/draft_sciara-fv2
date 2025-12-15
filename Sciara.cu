/**
 * Sciara.cu - CUDA Memory Allocation with Unified Memory
 *
 * This file replaces Sciara.cpp for CUDA versions.
 * Uses cudaMallocManaged for automatic CPU-GPU memory management.
 */

#include "Sciara.h"
#include "cal2DBuffer.h"
#include <cuda_runtime.h>
#include <stdio.h>

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

/**
 * Allocate substates using CUDA Unified Memory
 */
void allocateSubstates(Sciara *sciara)
{
    int size = sciara->domain->rows * sciara->domain->cols;
    int size_flows = size * NUMBER_OF_OUTFLOWS;

    // Allocate double arrays with Unified Memory
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sz_next, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Sh_next, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->ST_next, sizeof(double) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mf, sizeof(double) * size_flows));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mb, sizeof(bool) * size));
    CUDA_CHECK(cudaMallocManaged(&sciara->substates->Mhs, sizeof(double) * size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(sciara->substates->Sz, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->Sz_next, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->Sh, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->Sh_next, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->ST, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->ST_next, 0, sizeof(double) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->Mf, 0, sizeof(double) * size_flows));
    CUDA_CHECK(cudaMemset(sciara->substates->Mb, 0, sizeof(bool) * size));
    CUDA_CHECK(cudaMemset(sciara->substates->Mhs, 0, sizeof(double) * size));
}

/**
 * Deallocate substates - CUDA version
 */
void deallocateSubstates(Sciara *sciara)
{
    if (sciara->substates->Sz)      CUDA_CHECK(cudaFree(sciara->substates->Sz));
    if (sciara->substates->Sz_next) CUDA_CHECK(cudaFree(sciara->substates->Sz_next));
    if (sciara->substates->Sh)      CUDA_CHECK(cudaFree(sciara->substates->Sh));
    if (sciara->substates->Sh_next) CUDA_CHECK(cudaFree(sciara->substates->Sh_next));
    if (sciara->substates->ST)      CUDA_CHECK(cudaFree(sciara->substates->ST));
    if (sciara->substates->ST_next) CUDA_CHECK(cudaFree(sciara->substates->ST_next));
    if (sciara->substates->Mf)      CUDA_CHECK(cudaFree(sciara->substates->Mf));
    if (sciara->substates->Mb)      CUDA_CHECK(cudaFree(sciara->substates->Mb));
    if (sciara->substates->Mhs)     CUDA_CHECK(cudaFree(sciara->substates->Mhs));
}

/**
 * Evaluate power law parameters for viscosity and shear resistance
 */
void evaluatePowerLawParams(double PTvent, double PTsol, double value_sol, double value_vent, double &k1, double &k2)
{
    k2 = (log10(value_vent) - log10(value_sol)) / (PTvent - PTsol);
    k1 = log10(value_sol) - k2 * PTsol;
}

/**
 * Initialize simulation parameters
 */
void simulationInitialize(Sciara* sciara)
{
    unsigned int maximum_number_of_emissions = 0;

    // Reset the AC step
    sciara->simulation->step = 0;
    sciara->simulation->elapsed_time = 0;

    // Determine maximum number of steps from emissions
    for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
        if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
            maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();

    sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
    sciara->simulation->total_emitted_lava = 0;

    // Define morphology border
    makeBorder(sciara);

    // Compute power law parameters: a,b for viscosity, c,d for shear-resistance
    evaluatePowerLawParams(
        sciara->parameters->PTvent,
        sciara->parameters->PTsol,
        sciara->parameters->Pr_Tsol,
        sciara->parameters->Pr_Tvent,
        sciara->parameters->a,
        sciara->parameters->b);

    evaluatePowerLawParams(
        sciara->parameters->PTvent,
        sciara->parameters->PTsol,
        sciara->parameters->Phc_Tsol,
        sciara->parameters->Phc_Tvent,
        sciara->parameters->c,
        sciara->parameters->d);
}

// Moore neighborhood relative coordinates
int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1};
int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1};

/**
 * Initialize Sciara structure
 */
void init(Sciara*& sciara)
{
    sciara = new Sciara;
    sciara->domain = new Domain;

    sciara->X = new NeighsRelativeCoords;
    sciara->X->Xi = new int[MOORE_NEIGHBORS];
    sciara->X->Xj = new int[MOORE_NEIGHBORS];

    for (int n = 0; n < MOORE_NEIGHBORS; n++)
    {
        sciara->X->Xi[n] = _Xi[n];
        sciara->X->Xj[n] = _Xj[n];
    }

    sciara->substates = new Substates;
    sciara->parameters = new Parameters;
    sciara->simulation = new Simulation;
}

/**
 * Finalize and cleanup
 */
void finalize(Sciara*& sciara)
{
    deallocateSubstates(sciara);
    delete sciara->domain;
    delete[] sciara->X->Xi;
    delete[] sciara->X->Xj;
    delete sciara->X;
    delete sciara->substates;
    delete sciara->parameters;
    delete sciara->simulation;
    delete sciara;
    sciara = NULL;
}

/**
 * Mark boundary cells
 */
void makeBorder(Sciara *sciara)
{
    int j, i;
    int cols = sciara->domain->cols;
    int rows = sciara->domain->rows;

    // First row
    i = 0;
    for (j = 0; j < cols; j++)
        if (calGetMatrixElement(sciara->substates->Sz, cols, i, j) >= 0)
            calSetMatrixElement(sciara->substates->Mb, cols, i, j, true);

    // Last row
    i = rows - 1;
    for (j = 0; j < cols; j++)
        if (calGetMatrixElement(sciara->substates->Sz, cols, i, j) >= 0)
            calSetMatrixElement(sciara->substates->Mb, cols, i, j, true);

    // First column
    j = 0;
    for (i = 0; i < rows; i++)
        if (calGetMatrixElement(sciara->substates->Sz, cols, i, j) >= 0)
            calSetMatrixElement(sciara->substates->Mb, cols, i, j, true);

    // Last column
    j = cols - 1;
    for (i = 0; i < rows; i++)
        if (calGetMatrixElement(sciara->substates->Sz, cols, i, j) >= 0)
            calSetMatrixElement(sciara->substates->Mb, cols, i, j, true);

    // Interior cells adjacent to NODATA
    for (i = 1; i < rows - 1; i++)
        for (j = 1; j < cols - 1; j++)
            if (calGetMatrixElement(sciara->substates->Sz, cols, i, j) >= 0) {
                for (int k = 1; k < MOORE_NEIGHBORS; k++)
                    if (calGetMatrixElement(sciara->substates->Sz, cols,
                            i + sciara->X->Xi[k], j + sciara->X->Xj[k]) < 0) {
                        calSetMatrixElement(sciara->substates->Mb, cols, i, j, true);
                        break;
                    }
            }
}
