BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K1_PINNED_HBM := 0
K2_PINNED_HBM := 1
K2_TPB := 1024

.PHONY: all clean 

all: $(BIN_DIR)/lat_$(K1_PINNED_HBM)_bw_$(K2_PINNED_HBM)

$(BIN_DIR)/lat_$(K1_PINNED_HBM)_bw_$(K2_PINNED_HBM): main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DK1_PINNED_HBM=$(K1_PINNED_HBM) -DK2_PINNED_HBM=$(K2_PINNED_HBM) -DK2_TPB=$(K2_TPB) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/lat_*_bw_*