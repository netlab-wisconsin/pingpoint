# Build bw-ping-fidelity (E2) benchmark. Reuses main.h / k2.h home-identification from ../moe.
BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin
MOE_DIR  := $(BASE_DIR)/../moe

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:=
INCLUDES 	:= -I$(MOE_DIR) -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

TARGET := $(BIN_DIR)/bw_fidelity
DEPS   := bw_fidelity.cpp bw_fidelity.h $(MOE_DIR)/main.h $(MOE_DIR)/k2.h

.PHONY: all clean
all: $(TARGET)

$(BIN_DIR):
	mkdir -p $@

$(TARGET): $(DEPS) | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
