# Build MoE-serving FFN1 overhead-vs-ping-rate benchmark (E1).
BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin
MOE_DIR  := $(BASE_DIR)/../moe

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:=
INCLUDES 	:= -I$(MOE_DIR) -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

TARGET := $(BIN_DIR)/moe_serving

DEPS := moe_serving.cpp moe_serving.h \
        $(MOE_DIR)/main.h $(MOE_DIR)/ppnt.h $(MOE_DIR)/ppnt_only.h \
        $(MOE_DIR)/moe_only.h $(MOE_DIR)/k1.h $(MOE_DIR)/k2.h

.PHONY: all clean

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $@

$(TARGET): $(DEPS) | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
