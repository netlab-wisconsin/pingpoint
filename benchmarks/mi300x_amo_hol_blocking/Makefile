BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

# HIP_HOME 	:= /opt/rocm-7.1.0
HIP_HOME 	:= /opt/rocm-6.4.1
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= -fopenmp -O3
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

TPB   		:= 128
K1_MAX_TPB	:= 128
K2_MAX_TPB	:= 128

all: $(BIN_DIR)/amo_hol_blocking_$(TPB)_$(K1_MAX_TPB)_$(K2_MAX_TPB)

# fused bmk testing amo + bw
# amo_hol_blocking $(tpb) $(k1 max tpb) $(k2 max tpb)
$(BIN_DIR)/amo_hol_blocking_$(TPB)_$(K1_MAX_TPB)_$(K2_MAX_TPB): main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DTPB=$(TPB) -DK1_MAX_TPB=$(K1_MAX_TPB) -DK2_MAX_TPB=$(K2_MAX_TPB) \
	-o $@ $<

# histogram bmk testing only amo
$(BIN_DIR)/hist: hist.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/amo_hol_blocking_*
	rm -f $(BIN_DIR)/hist