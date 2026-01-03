BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K2_PINNED_XCD 	:= 1
K2_PINNED_HBM 	:= 3
K2_BPX_MIN 		:= 2	# Min nblocks per xcd when launching fused kernel k. Must be > 1.
K2_BPX_MAX 		:= 160	# Max nblocks per xcd when launching fused kernel k
K2_TPB 			:= 512

DEBUG_LEVEL := 0

# latency-sensitive kernels run at all links as 1 hop traffic
# bw-intensive kernel runs with cc(pinned xcd) -> cc(pinned hbm) traffic
all: $(BIN_DIR)/lat_all_all_bw_$(K2_PINNED_XCD)_$(K2_PINNED_HBM)_$(K2_TPB)

$(BIN_DIR)/lat_all_all_bw_$(K2_PINNED_XCD)_$(K2_PINNED_HBM)_$(K2_TPB): main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DK2_PINNED_XCD=$(K2_PINNED_XCD) -DK2_PINNED_HBM=$(K2_PINNED_HBM) \
	-DK2_BPX_MAX=$(K2_BPX_MAX) -DK2_BPX_MIN=$(K2_BPX_MIN) \
	-DK2_TPB=$(K2_TPB) \
	-DDEBUG_LEVEL=$(DEBUG_LEVEL) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/lat_all_all_bw_*