############
# COMPILER #
############

ifndef CPPC
	CPPC=g++
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

default:all

all:
	$(CPPC) *.cpp -o $(EXEC_SERIAL) -O0
	$(CPPC) *.cpp -o $(EXEC_OMP) -fopenmp -O0


#############
# EXECUTION #
#############

THREADS = 8
run_omp:
	OMP_NUM_THREADS=$(THREADS) ./$(EXEC_OMP) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)

run:
	./$(EXEC_SERIAL) $(INPUT_CONFIG) $(OUTPUT_CONFIG) $(STEPS) $(REDUCE_INTERVL) $(THICKNESS_THRESHOLD) && md5sum $(OUTPUT)


############
# CLEAN UP #
############

clean:
	rm -f $(EXEC_OMP) $(EXEC_SERIAL) *.o *output*

wipe:
	rm -f *.o *output*
