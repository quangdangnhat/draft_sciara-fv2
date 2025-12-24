# SCIARA-FV2 CUDA Implementation Report

## 1. Project Overview

This project implements the Sciara-fv2 lava flow simulation using CUDA with five different optimization strategies:
- **Global**: Baseline using only global memory
- **Tiled**: Shared memory tiling without halo
- **Tiled+Halo**: Shared memory with halo regions for complete neighbor access
- **CfAMe**: Conflict-free with atomic operations (memory-equivalent)
- **CfAMo**: Conflict-free with atomic operations (memory-optimized)

**Target GPU**: NVIDIA GeForce GTX 980 (Maxwell, Compute Capability 5.2)

---

## 2. Kernel Design & Tiled Selection Rationale

### 2.1 Kernel Classification by Access Pattern

| Kernel | Access Pattern | Data Reuse | Arithmetic Intensity | Tiled Benefit |
|--------|----------------|------------|---------------------|---------------|
| `emitLava` | Point access (vents only) | None | Very Low | **Low** - Few cells active |
| `computeOutflows` | Stencil (9-point Moore) | High | Medium | **High** - Neighbor reuse |
| `massBalance` | Stencil (9-point gather) | Medium | Low | **Medium** - Mf access dominates |
| `solidification` | Point access (self only) | Low | Medium | **Low** - No neighbor access |
| `reduceAdd` | Sequential reduction | High | Low | **High** - Shared memory reduction |

### 2.2 Tiled Kernel Selection Justification

#### `emitLava` - **NOT tiled in Global/CfA versions, Shared for vents in Tiled**

**Reasoning:**
- Only 2 vent cells are active out of 195,426 total cells (0.001%)
- Memory access is sparse and random (vent positions)
- Tiled version loads vent coordinates to shared memory for broadcast access
- **Decision**: Minimal benefit from full tiling; only vent data uses shared memory

```cpp
// Tiled version: Load vent coords to shared memory for all threads
extern __shared__ int s_vent_data[];
// All threads in block check same vent positions -> good broadcast
```

#### `computeOutflows` - **HIGH priority for tiling**

**Reasoning:**
- Each cell reads 9 neighbors (including self) → 9 global memory reads
- With 16×16 tile: interior cells (14×14 = 196) reuse neighbor data
- Border cells (60 cells) need global fallback without halo
- **Data reuse ratio**: Each loaded value used by up to 9 cells
- **Arithmetic intensity**: ~50 FLOPs per cell with multiple `pow()`, `atan()`, `sqrt()`

```
Memory access without tiling: 9 × 8 bytes × 195,426 cells = 14.0 MB
Memory access with tiling:    ~3 × 8 bytes × 195,426 cells = 4.7 MB (estimated)
Savings: ~66% reduction in global memory traffic
```

#### `massBalance` - **MEDIUM priority for tiling**

**Reasoning:**
- Reads neighbor temperatures from ST (9 reads)
- BUT also reads Mf buffer (8 × 2 = 16 accesses for in/out flows)
- Mf access pattern is strided (8 layers) → hard to tile efficiently
- **Decision**: Tile ST/Sh for neighbor access, Mf remains global

```
Mf layout: M[n*rows*cols + i*cols + j] where n ∈ [0,7]
Strided access prevents effective shared memory caching
```

#### `computeNewTemperatureAndSolidification` - **LOW priority for tiling**

**Reasoning:**
- Only reads own cell data (Sz, Sh, ST, Mb)
- No neighbor access required
- Compute-bound with `pow()` operations
- **Decision**: Tiled for coalesced loading but no neighbor sharing benefit

### 2.3 Why CfAMe/CfAMo Don't Use Tiling

**Atomic operations change the access pattern:**
- Traditional: Gather pattern (read from neighbors) → benefits from tiling
- CfAMe/CfAMo: Scatter pattern (write to neighbors) → tiling doesn't help

```cpp
// Scatter pattern - each cell writes to neighbors atomically
atomicAddDouble(&Sh_next[ni * c + nj], flow);
// Neighbors are random targets, not predictable loads
```

---

## 3. Warp Divergence Analysis

### 3.1 Divergence Points Identified

#### Point 1: Early exit in computeOutflows
```cpp
// sciara_fv2.cu:99-100
double h0 = GET(Sh, c, i, j);
if (h0 <= 0) return;  // DIVERGENCE POINT
```

**Impact Analysis:**
- Cells without lava: ~99% of grid initially, ~70% during simulation
- Threads exit early → remaining threads continue alone
- **Measured divergence**: Low impact because lava region is spatially coherent

**Countermeasure:**
- Considered: Active cell list to skip empty regions
- Decision: Not implemented due to overhead of maintaining dynamic list
- Spatial coherence of lava flow means warps are mostly uniform in active regions

#### Point 2: Minimization loop iterations
```cpp
// Variable iteration count per cell
do {
    loop = false;
    // ... elimination logic
    for (int k = 0; k < MOORE_NEIGHBORS; k++) {
        if (!eliminated[k] && avg <= H[k]) {
            eliminated[k] = true;
            loop = true;
        }
    }
} while (loop);  // DIVERGENCE: Different cells iterate different times
```

**Impact Analysis:**
- Most cells converge in 1-3 iterations
- Edge cases may need up to 8 iterations
- Within a warp: threads must wait for slowest

**Countermeasure:**
- Unrolling considered but increases register pressure significantly
- Current approach: Accept divergence, algorithmic correctness prioritized

#### Point 3: Boundary conditions
```cpp
if (ni < 0 || ni >= r || nj < 0 || nj >= c) {
    eliminated[k] = true;
    continue;  // DIVERGENCE: Boundary threads take different path
}
```

**Impact Analysis:**
- Only affects cells at domain boundaries (~0.5% of cells)
- Minimal performance impact due to small affected region

### 3.2 Divergence Measurements from Profiler

| Kernel | Branch Efficiency | Divergent Branches |
|--------|------------------|-------------------|
| computeOutflows | 85.2% | 14.8% |
| massBalance | 92.1% | 7.9% |
| solidification | 89.3% | 10.7% |

**Interpretation**: Moderate divergence but acceptable for stencil operations.

---

## 4. Grid/Block Size Exploration

### 4.1 Theoretical Analysis with CUDA Occupancy Calculator

**GTX 980 Specifications:**
- 16 SMs, 2048 threads/SM max, 64K registers/SM, 96 KB shared memory/SM
- Max blocks per SM: 32

**Register usage per kernel (from nvcc --ptxas-options=-v):**
- computeOutflows: ~48 registers/thread
- massBalance: ~32 registers/thread
- solidification: ~28 registers/thread

### 4.2 Block Size Comparison (Theoretical)

| Block Size | Threads | Blocks/SM | Occupancy | Notes |
|------------|---------|-----------|-----------|-------|
| 8×8 | 64 | 32 (max) | 100% | More blocks, higher latency hiding |
| 16×16 | 256 | 8 | 100% | **Current choice** - balanced |
| 32×32 | 1024 | 2 | 100% | Fewer blocks, less flexibility |
| 32×8 | 256 | 8 | 100% | Better for row-oriented access |
| 16×8 | 128 | 16 | 100% | Good for smaller grids |

### 4.3 Shared Memory Constraints

| Version | Shared Memory/Block | Max Blocks/SM | Limiting Factor |
|---------|--------------------|--------------|-|
| Tiled (16×16) | 3 × 256 × 8 = 6 KB | 16 | Not limited |
| Tiled+Halo (18×18) | 3 × 324 × 8 = 7.8 KB | 12 | Not limited |

### 4.4 Recommended Block Sizes

Based on analysis:
- **computeOutflows**: 16×16 optimal (neighbor access pattern)
- **massBalance**: 16×16 or 32×8 (row coalescing)
- **solidification**: 32×8 or 16×16 (element-wise)
- **reduceAdd**: 256×1 (standard reduction)

**Recommendation**: Use 16×16 uniformly for simplicity and good occupancy.

To run block exploration:
```bash
nvcc -O3 -arch=sm_52 block_size_exploration.cu -o block_explore
./block_explore
```

---

## 5. Performance Analysis

### 5.1 Per-Kernel Execution Times (16,000 steps)

#### Global Version
| Kernel | Time (s) | Calls | Avg (μs) | % Total |
|--------|----------|-------|----------|---------|
| massBalance | 2.060 | 16,000 | 128.7 | 70.3% |
| computeOutflows | 0.603 | 16,000 | 37.7 | 20.6% |
| solidification | 0.265 | 16,000 | 16.6 | 9.1% |
| reduceAdd | 0.0004 | 32 | 11.8 | 0.01% |
| **Total GPU** | **2.928** | | | |

#### Tiled Version
| Kernel | Time (s) | Calls | Avg (μs) | % Total |
|--------|----------|-------|----------|---------|
| massBalance_tiled | 2.129 | 16,000 | 133.0 | 62.6% |
| computeOutflows_tiled | 0.774 | 16,000 | 48.4 | 22.8% |
| solidification_tiled | 0.497 | 16,000 | 31.1 | 14.6% |
| reduceAdd | 0.0004 | 32 | 12.3 | 0.01% |
| **Total GPU** | **3.400** | | | |

#### CfAMe Version
| Kernel | Time (s) | Calls | Avg (μs) | % Total |
|--------|----------|-------|----------|---------|
| CfA_Me | 0.878 | 16,000 | 54.9 | 46.5% |
| initBuffers | 0.537 | 16,000 | 33.5 | 28.4% |
| solidification | 0.241 | 16,000 | 15.0 | 12.8% |
| normalizeTemp | 0.232 | 16,000 | 14.5 | 12.3% |
| **Total GPU** | **1.888** | | | |

### 5.2 Speedup Analysis (vs Global baseline)

| Version | Total Time (s) | Speedup | Analysis |
|---------|----------------|---------|----------|
| Global | 8.367 | 1.00× | Baseline |
| Tiled | 10.916 | 0.77× | **Slower** - sync overhead |
| Tiled+Halo | 9.311 | 0.90× | **Slower** - halo loading cost |
| CfAMe | 7.628 | 1.10× | Faster - fewer kernels |
| CfAMo | 7.239 | **1.16×** | **Fastest** - no Mf buffer |

**Why Tiled versions are slower:**
1. Grid size (517×378) is small - shared memory overhead > benefit
2. Shared memory synchronization (`__syncthreads()`) adds latency
3. Halo loading requires multiple passes per thread
4. Global memory is fast on GTX 980 with good L2 cache

**Why CfAMo is fastest:**
1. Eliminates Mf buffer (saves 12.5 MB memory)
2. Merges computeOutflows + massBalance → fewer kernel launches
3. Atomic operations have acceptable contention for sparse lava cells
4. Better cache utilization with smaller memory footprint

### 5.3 Occupancy Comparison

| Version | computeOutflows | massBalance | solidification |
|---------|-----------------|-------------|----------------|
| Global | 26.4% | 89.8% | 75.7% |
| Tiled | 31.2% | 85.4% | 72.3% |
| Tiled+Halo | 28.7% | 82.1% | 69.8% |
| CfAMe | 18.5% (CfA_Me) | - | 75.7% |
| CfAMo | 18.5% (CfA_Mo) | - | 75.7% |

**Note**: CfAMe/CfAMo have lower occupancy due to higher register usage in combined kernel.

---

## 6. Roofline Analysis

### 6.1 Hardware Roofline (GTX 980)

**Peak Performance:**
- FP64: 155.7 GFLOP/s (1/32 of FP32)
- FP32: 4981.0 GFLOP/s

**Memory Bandwidth:**
- Global Memory: 224.3 GB/s
- Shared Memory: 2119.7 GB/s
- L2 Cache: ~400 GB/s (estimated)

**Ridge Points:**
- Global: 155.7 / 224.3 = **0.694 FLOP/Byte**
- Shared: 155.7 / 2119.7 = **0.073 FLOP/Byte**

### 6.2 Arithmetic Intensity Calculation

#### computeOutflows Kernel (Manual Analysis)

**FLOPs per cell (approximate):**
```
- pow(10, a+b*T): 2 FLOPs (exponent) + 20 FLOPs (pow) = 22
- pow(10, c+d*T): 22 FLOPs
- sqrt(2.0): 10 FLOPs (once)
- atan(): 20 FLOPs × 8 neighbors = 160
- cos(): 10 FLOPs × 8 = 80
- Arithmetic: ~50 FLOPs
- Total: ~350 FLOPs (FP64)
```

**Bytes per cell:**
```
Reads:
- Sz[9]: 9 × 8 = 72 bytes
- Sh[9]: 9 × 8 = 72 bytes
- ST[1]: 8 bytes
Writes:
- Mf[8]: 8 × 8 = 64 bytes
Total: 216 bytes
```

**AI = 350 / 216 = 1.62 FLOP/Byte** (theoretical)

**Measured AI from profiler: ~0.041 FLOP/Byte**

**Discrepancy explanation:**
- Profiler measures actual memory transactions, not unique bytes
- Cache misses cause redundant memory traffic
- Unified Memory adds overhead

#### massBalance Kernel

**FLOPs per cell:**
```
- 8 × (add, multiply): 16 FLOPs
- Division: 20 FLOPs
- Total: ~36 FLOPs
```

**Bytes per cell:**
```
- Sh, ST, Mf reads: ~200 bytes
- Sh_next, ST_next writes: 16 bytes
```

**AI = 36 / 216 = 0.17 FLOP/Byte** → Memory-bound

### 6.3 Kernel Placement on Roofline

```
                    Ridge Point (0.694)
                          |
GFLOPS                    v
   ^
155|--------------------+========  Peak FP64
   |                   /
   |                  /
 35|        ★ Global /
   |       ★ Tiled  /
   |      ★ Halo   /
   |             /
  1|     ★ CfA* /
   +-----|-----|------|-------> AI (FLOP/Byte)
       0.001 0.04   0.69
```

**Analysis:**
- All kernels are **MEMORY-BOUND** (AI < 0.694)
- Global/Tiled/Halo at AI ≈ 0.04 → achieving ~20% of memory bandwidth ceiling
- CfAMe/CfAMo at AI ≈ 0.0002 → atomic operations dominate

### 6.4 Roofline Interpretation

| Version | AI | Performance | Bound | Optimization Potential |
|---------|-----|-------------|-------|----------------------|
| Global | 0.041 | 35.8 GFLOP/s | Memory | Improve coalescing |
| Tiled | 0.043 | 32.9 GFLOP/s | Memory | Reduce sync overhead |
| Tiled+Halo | 0.045 | 30.1 GFLOP/s | Memory | Halo loading efficient |
| CfAMe | 0.0002 | 0.02 GFLOP/s | Atomic | Reduce contention |
| CfAMo | 0.0002 | 0.02 GFLOP/s | Atomic | Reduce contention |

**Key Findings:**
1. All versions are memory-bound, not compute-bound
2. Tiling improves AI slightly but overhead exceeds benefit
3. CfA* versions have artificially low AI due to atomic operations
4. Actual performance determined by memory system efficiency

---

## 7. Floating-Point Numerical Accuracy

### 7.1 Sources of Numerical Variation

1. **Parallel reduction order**: Sum order affects FP rounding
2. **Atomic operations**: Non-deterministic execution order
3. **Compiler optimizations**: FMA (fused multiply-add) can change results

### 7.2 Mitigation Strategies

```cpp
// Compile without FMA for reproducibility:
// nvcc -fmad=false ...

// Current setting in Makefile:
NVFLAGS=-O3 -arch=sm_52  // FMA enabled (default)
```

### 7.3 Expected Differences

- CfAMe/CfAMo may produce slightly different results than Global/Tiled
- Differences in the range of 1e-10 to 1e-14 (FP64 precision)
- Results remain physically valid (no NaN, no explosion)

---

## 8. Conclusions & Recommendations

### 8.1 Summary

1. **Best performing version**: CfAMo (1.16× speedup)
2. **Tiling overhead**: Not beneficial for this grid size
3. **All versions**: Memory-bound operation
4. **Atomic operations**: Acceptable for sparse active cells

### 8.2 Recommendations for Larger Grids

If using larger grids (e.g., 2048×2048):
- Tiled versions may become beneficial
- Consider 32×32 block size
- Implement persistent kernels for reduced launch overhead

### 8.3 Future Optimizations

1. **Active cell list**: Skip empty regions entirely
2. **Stream-based execution**: Overlap kernel execution
3. **Multi-GPU**: Domain decomposition for very large grids
4. **Mixed precision**: FP32 for non-critical computations

---

## Appendix A: Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `Sciara.cu` | CUDA memory allocation | cudaMallocManaged for all substates |
| `sciara_fv2.cu` | Global version | Baseline CUDA kernels |
| `sciara_fv2_tiled.cu` | Tiled version | Shared memory for tiles |
| `sciara_fv2_tiled_halo.cu` | Halo version | 18×18 shared memory with halo |
| `sciara_fv2_cfame.cu` | CfAMe version | Atomic operations, keeps Mf |
| `sciara_fv2_cfamo.cu` | CfAMo version | Atomic operations, no Mf |

## Appendix B: Build & Run Commands

```bash
# Build all CUDA versions
make sciara_cuda

# Run benchmark
make benchmark

# Run profiling
make profile

# Block size exploration
nvcc -O3 -arch=sm_52 block_size_exploration.cu -o block_explore
./block_explore
```

## Appendix C: Profiling Data Files

Located in `profiling_results/`:
- `*_gpu_summary.csv`: Kernel execution times
- `*_compute.csv`: FLOP counts
- `*_memory.csv`: Memory throughput
- `*_occupancy.csv`: Achieved occupancy
- `roofline_fp64.png`: Roofline plot
- `histogram_times.png`: Execution time comparison
- `occupancy.png`: Occupancy comparison

---

*Report generated for Sciara-fv2 CUDA Project*
*GPU: NVIDIA GeForce GTX 980 (Compute Capability 5.2)*
*Dataset: Mt. Etna 2006 (517×378 cells, 16,000 steps)*
