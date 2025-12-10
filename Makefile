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
EXEC_TILED = sciara_tiled
EXEC_HALO = sciara_halo
EXEC_CFAME = sciara_cfame
EXEC_CFAMO = sciara_cfamo

default:all

all:
	$(CPPC) *.cpp -o $(EXEC_SERIAL) -O0
	$(CPPC) *.cpp -o $(EXEC_OMP) -fopenmp -O0

cuda:
	$(NVCC) sciara_cuda.cu sciara_fv2_cuda.cu io.cpp GISInfo.cpp vent.cpp configurationPathLib.cpp cal2DBuffer.cpp cal2DBufferIO.cpp -o $(EXEC_CUDA) -O3 -arch=sm_52

tiled:
	$(NVCC) sciara_cuda.cu sciara_fv2_tiled.cu io.cpp GISInfo.cpp vent.cpp configurationPathLib.cpp cal2DBuffer.cpp cal2DBufferIO.cpp -o $(EXEC_TILED) -O3 -arch=sm_52

halo:
	$(NVCC) sciara_cuda.cu sciara_fv2_halo.cu io.cpp GISInfo.cpp vent.cpp configurationPathLib.cpp cal2DBuffer.cpp cal2DBufferIO.cpp -o $(EXEC_HALO) -O3 -arch=sm_52

cfame:
	$(NVCC) sciara_cuda.cu sciara_fv2_cfame.cu io.cpp GISInfo.cpp vent.cpp configurationPathLib.cpp cal2DBuffer.cpp cal2DBufferIO.cpp -o $(EXEC_CFAME) -O3 -arch=sm_52

cfamo:
	$(NVCC) sciara_cuda.cu sciara_fv2_cfamo.cu io.cpp GISInfo.cpp vent.cpp configurationPathLib.cpp cal2DBuffer.cpp cal2DBufferIO.cpp -o $(EXEC_CFAMO) -O3 -arch=sm_52


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

run_tiled:
	./$(EXEC_TILED) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_halo:
	./$(EXEC_HALO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cfame:
	./$(EXEC_CFAME) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run_cfamo:
	./$(EXEC_CFAMO) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)


############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_OMP) $(EXEC_SERIAL) $(EXEC_CUDA) $(EXEC_TILED) $(EXEC_CFAME) $(EXEC_CFAMO) *.o *output*

wipe:
	rm -f *.o *output*
