BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

K2_BPX_MAX := 40 # TODO: set proper value

.PHONY: all clean

all: $(BIN_DIR)/stateful_routing

$(BIN_DIR)/stateful_routing: main.cpp
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DK2_BPX_MAX=$(K2_BPX_MAX) -o $@ $<

clean: 
	rm -f $(BIN_DIR)/stateful_routing