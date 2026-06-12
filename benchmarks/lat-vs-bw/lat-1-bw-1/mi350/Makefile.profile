BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx950
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -lhsa-runtime64 -lrocm_smi64 -ldl

K1_PINNED_XCD := 0
K1_PINNED_CC := 0

K2_PINNED_XCD := 1
K2_PINNED_CC := 1

# Min nblocks per xcd when launching fused kernel k.
K2_BPX_MIN := 1
# Max nblocks per xcd when launching fused kernel k
K2_BPX_MAX := 76
K2_TPB := 512

SUFFIX := prof_gfx950

# PROFILE specific flags
CCFLAGS += -DUSE_GLOBAL_BARRIER=1 # cooperative groups conflict with rocprof
CCFLAGS += -DEPOCHS=1 # reduce the number of epochs for profiling
CCFLAGS += -DPROFILE=1

TARGET := $(BIN_DIR)/bpx_$(K2_BPX_MIN)_$(K2_BPX_MAX)_tpb_$(K2_TPB)_$(SUFFIX)

all: $(TARGET)

$(TARGET): main.cpp main.h k1.h k2.h  
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DK1_PINNED_XCD=$(K1_PINNED_XCD) -DK2_PINNED_XCD=$(K2_PINNED_XCD) \
	-DK1_PINNED_CC=$(K1_PINNED_CC) -DK2_PINNED_CC=$(K2_PINNED_CC) \
	-DK2_BPX_MIN=$(K2_BPX_MIN) -DK2_BPX_MAX=$(K2_BPX_MAX) \
	-DK2_TPB=$(K2_TPB) \
	-DSKIP_HOME_IDENTIFICATION=1 \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/*_$(SUFFIX)
