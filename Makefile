############
# COMPILER #
############

ifndef CPPC
    CPPC=g++
endif

NVCC=nvcc
# GTX 980 = Compute Capability 5.2
NVFLAGS=-O3 -arch=sm_52 # -fmad=false to disable fused multiply-add

###########
# DATASET #
###########

INPUT_CONFIG=./data/2006/2006_000000000000.cfg
OUTPUT_CONFIG=./data/2006/output_2006
OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc ./data/2006/output_2006_000000016000_EmissionRate.txt ./data/2006/output_2006_000000016000_SolidifiedLavaThickness.asc ./data/2006/output_2006_000000016000_Morphology.asc ./data/2006/output_2006_000000016000_Vents.asc ./data/2006/output_2006_000000016000_Thickness.asc 
STEPS=16000
REDUCE_INTERVL=1000
THICKNESS_THRESHOLD=1.0


###############################
# COMPILATION #
###############################

# Run ./data/2006/2006_000000000000.cfg ./data/2006_OUT/output_2006 16000 1000 1.0

EXEC_OMP = sciara_omp
EXEC_SERIAL = sciara_serial
EXEC_CUDA = sciara_cuda
EXEC_CUDA_TILED = sciara_cuda_tiled
EXEC_CUDA_TILED_HALO = sciara_cuda_tiled_halo
EXEC_CUDA_CFAME = sciara_cuda_cfame
EXEC_CUDA_CFAMO = sciara_cuda_cfamo

ALL_CPPS = $(wildcard *.cpp)
CPP_SOURCES_FOR_CUDA = $(filter-out sciara_fv2.cpp Sciara.cpp, $(ALL_CPPS))

default: all

all: serial omp sciara_cuda

serial:
	$(CPPC) $(ALL_CPPS) -o $(EXEC_SERIAL) -O3

omp:
	$(CPPC) $(ALL_CPPS) -o $(EXEC_OMP) -fopenmp -O3

cuda:
	$(NVCC) $(NVFLAGS) $(CPP_SOURCES_FOR_CUDA) sciara_fv2.cu Sciara.cu -o $(EXEC_CUDA)

cuda_tiled:
	$(NVCC) $(NVFLAGS) $(CPP_SOURCES_FOR_CUDA) sciara_fv2_tiled.cu Sciara.cu -o $(EXEC_CUDA_TILED)

cuda_tiled_halo:
	$(NVCC) $(NVFLAGS) $(CPP_SOURCES_FOR_CUDA) sciara_fv2_tiled_halo.cu Sciara.cu -o $(EXEC_CUDA_TILED_HALO)

cuda_cfame:
	$(NVCC) $(NVFLAGS) $(CPP_SOURCES_FOR_CUDA) sciara_fv2_cfame.cu Sciara.cu -o $(EXEC_CUDA_CFAME)

cuda_cfamo:
	$(NVCC) $(NVFLAGS) $(CPP_SOURCES_FOR_CUDA) sciara_fv2_cfamo.cu Sciara.cu -o $(EXEC_CUDA_CFAMO)

sciara_cuda: cuda cuda_tiled cuda_tiled_halo cuda_cfame cuda_cfamo


#############
# EXECUTION #
#############

THREADS = 8

run:
	./$(EXEC_SERIAL) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_omp:
	OMP_NUM_THREADS=$(THREADS) ./$(EXEC_OMP) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda:
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_tiled:
	./$(EXEC_CUDA_TILED) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_tiled_halo:
	./$(EXEC_CUDA_TILED_HALO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_cfame:
	./$(EXEC_CUDA_CFAME) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda_cfamo:
	./$(EXEC_CUDA_CFAMO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

# Run all CUDA versions
run_all_cuda: sciara_cuda
	@echo "============================================"
	@echo "Running CUDA Global Version..."
	@echo "============================================"
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD)
	@echo ""
	@echo "============================================"
	@echo "Running CUDA Tiled Version..."
	@echo "============================================"
	./$(EXEC_CUDA_TILED) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD)
	@echo ""
	@echo "============================================"
	@echo "Running CUDA Tiled+Halo Version..."
	@echo "============================================"
	./$(EXEC_CUDA_TILED_HALO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD)
	@echo ""
	@echo "============================================"
	@echo "Running CUDA CfAMe Version..."
	@echo "============================================"
	./$(EXEC_CUDA_CFAME) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD)
	@echo ""
	@echo "============================================"
	@echo "Running CUDA CfAMo Version..."
	@echo "============================================"
	./$(EXEC_CUDA_CFAMO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD)

# Verify correctness (md5 checksum)
EXPECTED_MD5=704a4a65d1890589e952b155d53b110d
TEMP_OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc

verify: sciara_cuda
	@echo "Verifying CUDA Global..."
	@./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) > /dev/null 2>&1
	@md5sum $(TEMP_OUTPUT) | grep -q $(EXPECTED_MD5) && echo "CUDA Global: PASS" || echo "CUDA Global: FAIL"
	@echo "Verifying CUDA Tiled..."
	@./$(EXEC_CUDA_TILED) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) > /dev/null 2>&1
	@md5sum $(TEMP_OUTPUT) | grep -q $(EXPECTED_MD5) && echo "CUDA Tiled: PASS" || echo "CUDA Tiled: FAIL"
	@echo "Verifying CUDA Tiled+Halo..."
	@./$(EXEC_CUDA_TILED_HALO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) > /dev/null 2>&1
	@md5sum $(TEMP_OUTPUT) | grep -q $(EXPECTED_MD5) && echo "CUDA Tiled+Halo: PASS" || echo "CUDA Tiled+Halo: FAIL"
	@echo "Verifying CUDA CfAMe..."
	@./$(EXEC_CUDA_CFAME) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) > /dev/null 2>&1
	@md5sum $(TEMP_OUTPUT) | grep -q $(EXPECTED_MD5) && echo "CUDA CfAMe: PASS" || echo "CUDA CfAMe: FAIL"
	@echo "Verifying CUDA CfAMo..."
	@./$(EXEC_CUDA_CFAMO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) > /dev/null 2>&1
	@md5sum $(TEMP_OUTPUT) | grep -q $(EXPECTED_MD5) && echo "CUDA CfAMo: PASS" || echo "CUDA CfAMo: FAIL"

############
#  PROFILE #
############

profile: sciara_cuda
	chmod +x run_profiling.sh
	./run_profiling.sh

############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_OMP) $(EXEC_SERIAL) $(EXEC_CUDA) $(EXEC_CUDA_TILED) $(EXEC_CUDA_TILED_HALO) $(EXEC_CUDA_CFAME) $(EXEC_CUDA_CFAMO) *.o *out*

wipe:
	rm -f *.o *out*

clean-profile:
	rm -rf profiling_results/

clean-all: clean wipe clean-profile

.PHONY: default all serial omp cuda cuda_tiled cuda_tiled_halo cuda_cfame cuda_cfamo sciara_cuda run run_omp run_cuda run_cuda_tiled run_cuda_tiled_halo run_cuda_cfame run_cuda_cfamo run_all_cuda verify clean wipe clean-profile clean-all profile
