BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K1_PINNED_XCD := 0
K1_PINNED_HBM := 2
K2_PINNED_XCD := 1
K2_PINNED_HBM := 3
K2_BPX_MAX := 160

.PHONY: all clean 

all: $(BIN_DIR)/stateful_routing_$(K1_PINNED_XCD)$(K1_PINNED_HBM)$(K2_PINNED_XCD)$(K2_PINNED_HBM)

$(BIN_DIR)/stateful_routing_$(K1_PINNED_XCD)$(K1_PINNED_HBM)$(K2_PINNED_XCD)$(K2_PINNED_HBM): main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DK1_PINNED_XCD=$(K1_PINNED_XCD) -DK1_PINNED_HBM=$(K1_PINNED_HBM) \
	-DK2_PINNED_XCD=$(K2_PINNED_XCD) -DK2_PINNED_HBM=$(K2_PINNED_HBM) -DK2_BPX_MAX=$(K2_BPX_MAX) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/stateful_routing*