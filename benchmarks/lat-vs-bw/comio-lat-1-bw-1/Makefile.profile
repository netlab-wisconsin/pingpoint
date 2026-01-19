BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K1_PINNED_HBM := 0
K2_PINNED_HBM := 1
K2_TPB := 1024
K2_BPX_MIN := 1
K2_BPX_MAX := 1

# PROFILE specific flags
CCFLAGS += -DUSE_GLOBAL_BARRIER=1 # cooperative groups conflict with rocprof
CCFLAGS += -DEPOCHS=1 # reduce the number of epochs for profiling
CCFLAGS += -DPROFILE=1

SUFFIX := prof

all: $(BIN_DIR)/lat_$(K1_PINNED_HBM)_bw_$(K2_PINNED_HBM)_bpx_$(K2_BPX_MIN)_$(K2_BPX_MAX)_$(SUFFIX)

$(BIN_DIR)/lat_$(K1_PINNED_HBM)_bw_$(K2_PINNED_HBM)_bpx_$(K2_BPX_MIN)_$(K2_BPX_MAX)_$(SUFFIX): main.cpp main.h k1.h k2.h 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DK1_PINNED_HBM=$(K1_PINNED_HBM) -DK2_PINNED_HBM=$(K2_PINNED_HBM) -DK2_TPB=$(K2_TPB) \
	-DK2_BPX_MIN=$(K2_BPX_MIN) -DK2_BPX_MAX=$(K2_BPX_MAX) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/lat_*_bw_*_bpx_*_*_$(SUFFIX)