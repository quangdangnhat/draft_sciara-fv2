#!/bin/bash

# Configuration
INPUT_CONFIG="./data/2006/2006_000000000000.cfg"
OUTPUT_CONFIG="./data/2006/output_2006"
STEPS=16000
REDUCED_STEPS=10
REDUCED_STEPS_INTERVAL=10
REDUCE_INTERVAL=1000
THICKNESS_THRESHOLD=1.0

OUTPUT_PROFILE="./profiling_results"
mkdir -p "$OUTPUT_PROFILE"

# Find executables
EXECUTABLES=$(find . -maxdepth 1 -type f -name "*cuda*" ! -name "*.*" -executable)

echo "=========================================================="
echo " [1/4] RUNNING GPUMEMBENCH (Microbenchmark)"
echo "=========================================================="

# Check if gpumembench exists, if not, try to compile it from source
if [ ! -f ./gpumembench ]; then
    if [ -f ./gpumembench.cu ]; then
        echo "Compiling gpumembench..."
        nvcc -arch=sm_52 -O3 gpumembench.cu -o gpumembench
    else
        echo "WARNING: gpumembench not found! Using default GTX 980 values."
    fi
fi

if [ -f ./gpumembench ]; then
    ./gpumembench > "$OUTPUT_PROFILE/gpumembench.log"
    echo "Microbenchmark results saved."
else
    # Create dummy log with default GTX 980 values if binary missing
    echo "Global read: 224.3 GB/s" > "$OUTPUT_PROFILE/gpumembench.log"
    echo "Shared read: 2119.7 GB/s" >> "$OUTPUT_PROFILE/gpumembench.log"
    echo "Texture read: 28008.71 GB/s" >> "$OUTPUT_PROFILE/gpumembench.log" # Approx for L1/Tex
fi

echo "=========================================================="
echo " [2/4] BENCHMARK: Measuring ACTUAL Execution Time (NO nvprof)"
echo "=========================================================="

# Sort executables for consistent order (matches Makefile order)
SORTED_EXECS=$(echo "$EXECUTABLES" | tr ' ' '\n' | sort -r | tr '\n' ' ')

echo "Running each executable WITHOUT profiler overhead..."
echo "Execution order: $(echo $SORTED_EXECS | tr '\n' ' ')"

# GPU Warmup: Run a short simulation to stabilize GPU clocks/thermals
echo "  [Warmup] Stabilizing GPU state..."
first_exe=$(echo $SORTED_EXECS | awk '{print $1}')
if [ -n "$first_exe" ]; then
    $first_exe $INPUT_CONFIG $OUTPUT_CONFIG 100 100 $THICKNESS_THRESHOLD > /dev/null 2>&1
fi

# Actual benchmark runs
for exe in $SORTED_EXECS; do
    exe_name=$(basename "$exe")
    echo "  Benchmarking: $exe_name"
    # Run WITHOUT nvprof to get actual wall-clock time
    ./$exe_name $INPUT_CONFIG $OUTPUT_CONFIG $STEPS $REDUCE_INTERVAL $THICKNESS_THRESHOLD > "${OUTPUT_PROFILE}/${exe_name}_benchmark.log" 2>&1
done
echo "Benchmark complete. Times saved to *_benchmark.log files."

echo "=========================================================="
echo " [3/4] PROFILING: Collecting GPU Metrics (with nvprof)"
echo "=========================================================="

echo "Executables to be profiled (same order as benchmark):"
echo "$SORTED_EXECS"

for exe in $SORTED_EXECS; do
    exe_name=$(basename "$exe")

    echo ""
    echo "----------------------------------------------------------"
    echo " Profiling: $exe_name"
    echo "----------------------------------------------------------"

    # 1. GPU Summary (kernel times - for roofline calculation)
    echo "[1/4] Collecting GPU Summary..."
    nvprof --print-gpu-summary --log-file "${OUTPUT_PROFILE}/${exe_name}_gpu_summary.csv" --csv \
        ./$exe_name $INPUT_CONFIG $OUTPUT_CONFIG $STEPS $REDUCE_INTERVAL $THICKNESS_THRESHOLD > "${OUTPUT_PROFILE}/${exe_name}.log" 2>&1

    # 2. Compute Metrics (FP64, FP32, FP16 FLOP counts)
    echo "[2/4] Collecting Compute Metrics (FLOP counts)..."
    nvprof --metrics flop_count_dp,flop_count_sp,flop_count_hp --log-file "${OUTPUT_PROFILE}/${exe_name}_compute.csv" --csv \
        ./$exe_name $INPUT_CONFIG $OUTPUT_CONFIG $REDUCED_STEPS $REDUCED_STEPS_INTERVAL $THICKNESS_THRESHOLD > /dev/null 2>&1

    # 3. Memory Hierarchy: Transaction Counts
    echo "[3/4] Collecting Memory Metrics (Transaction Counts)..."
    nvprof --metrics gld_transactions,gst_transactions,atomic_transactions,local_load_transactions,local_store_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions \
        --log-file "${OUTPUT_PROFILE}/${exe_name}_memory.csv" --csv \
        ./$exe_name $INPUT_CONFIG $OUTPUT_CONFIG $REDUCED_STEPS $REDUCED_STEPS_INTERVAL $THICKNESS_THRESHOLD > /dev/null 2>&1

    # 4. Occupancy (Achieved Occupancy)
    echo "[4/4] Collecting Occupancy Metric (achieved_occupancy)..."
    nvprof --metrics achieved_occupancy \
        --log-file "${OUTPUT_PROFILE}/${exe_name}_occupancy.csv" --csv \
        ./$exe_name $INPUT_CONFIG $OUTPUT_CONFIG $REDUCED_STEPS $REDUCED_STEPS_INTERVAL $THICKNESS_THRESHOLD > /dev/null 2>&1
    echo "      -> Done. Logs saved to ${exe_name}_*.csv"
done

echo "========================================================="
echo " [4/4] PARSING METRICS AND PLOTTING RESULTS"
echo "========================================================="

python3 parse_metrics.py
gnuplot plot_roofline.gp
gnuplot plot_histogram.gp
gnuplot plot_occupancy.gp

echo " Results saved in ${OUTPUT_PROFILE}/"
