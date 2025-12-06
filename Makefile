############
# COMPILER #
############

ifndef CPPC
	CPPC=g++
endif

ifndef NVCC
	NVCC=nvcc
endif


###########
# DATASET #
###########

INPUT_CONFIG=./data/2006/2006_000000000000.cfg
OUTPUT_CONFIG=./data/2006/output_2006
OUTPUT=./data/2006/output_2006_000000016000_Temperature.asc #md5sum: 0c071cd864046d3c6aaf30997290ad6c
STEPS=1000
REDUCE_INTERVL=1000
THICKNESS_THRESHOLD=1.0#resulting in 16000 steps


###############################
# VIM'S TERMDEBUG RUN COMMAND #
###############################

# Run ./data/2006/2006_000000000000.cfg ./data/2006_OUT/output_2006 1000 1000 1.0


###############
# COMPILATION #
###############

EXEC_OMP = sciara_omp
EXEC_SERIAL = sciara_serial
EXEC_CUDA = sciara_cuda

# CUDA architecture (adjust based on your GPU)
# sm_52 for GTX 980, sm_61 for GTX 1080, sm_75 for RTX 2080, sm_86 for RTX 3080
CUDA_ARCH = -arch=sm_50

# Source files
CPP_SOURCES = GISInfo.cpp cal2DBuffer.cpp cal2DBufferIO.cpp configurationPathLib.cpp io.cpp Sciara.cpp vent.cpp
CUDA_SOURCES = sciara_fv2_cuda.cu

# CUDA flags
NVCC_FLAGS = -O3 $(CUDA_ARCH) -Xcompiler -O3

default:all

all: serial omp

serial:
	$(CPPC) *.cpp -o $(EXEC_SERIAL) -O0

omp:
	$(CPPC) *.cpp -o $(EXEC_OMP) -fopenmp -O0

cuda:
	$(NVCC) $(NVCC_FLAGS) $(CPP_SOURCES) $(CUDA_SOURCES) -o $(EXEC_CUDA)

cuda_debug:
	$(NVCC) -g -G $(CUDA_ARCH) $(CPP_SOURCES) $(CUDA_SOURCES) -o $(EXEC_CUDA)


#############
# EXECUTION #
#############

THREADS = 8
run_omp:
	OMP_NUM_THREADS=$(THREADS) ./$(EXEC_OMP) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run:
	./$(EXEC_SERIAL) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cuda:
	./$(EXEC_CUDA) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)


############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_OMP) $(EXEC_SERIAL) $(EXEC_CUDA) *.o *output*

wipe:
	rm -f *.o *output*

clean_output:
	rm -f ./data/2006/output_*
