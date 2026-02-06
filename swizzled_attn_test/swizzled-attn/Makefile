BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K1_PINNED_XCD := 0
K1_PINNED_HBM := 0

K2_PINNED_XCD := 1
K2_PINNED_HBM := 1

# Min nblocks per xcd when launching fused kernel k.
K2_BPX_MIN := 1
# Max nblocks per xcd when launching fused kernel k
K2_BPX_MAX := 80
K2_TPB := 1024

SUFFIX := run

# TARGET := $(BIN_DIR)/lat_xcd_$(K1_PINNED_XCD)_hbm_$(K1_PINNED_HBM)_bw_xcd_$(K2_PINNED_XCD)_hbm_$(K2_PINNED_HBM)_bpx_$(K2_BPX_MIN)_$(K2_BPX_MAX)_tpb_$(K2_TPB)_$(SUFFIX)
TARGET := $(BIN_DIR)/attn 

all: $(TARGET)

$(TARGET): main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

clean: 
	rm -f $(TARGET)