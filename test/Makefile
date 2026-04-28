# Build MoE HIP test binaries ("all" vs "only")
BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

MOE_ALL_TARGET  := $(BIN_DIR)/moe_all
MOE_ONLY_TARGET := $(BIN_DIR)/moe_only

.PHONY: all only clean

all: $(MOE_ALL_TARGET)
only: $(MOE_ONLY_TARGET)

$(BIN_DIR):
	mkdir -p $@

$(MOE_ALL_TARGET): moe.cpp moe.h ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

$(MOE_ONLY_TARGET): moe_only.cpp moe_only.h ppnt_only.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

clean:
	rm -f $(MOE_ALL_TARGET) $(MOE_ONLY_TARGET)
