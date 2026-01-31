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
# SUFFIX := notgt

TARGET := $(BIN_DIR)/moe_$(SUFFIX)

all: $(TARGET)

$(TARGET): moe.cpp moe.h ppnt.h main.h k1.h k2.h
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

# $(BIN_DIR)/attn: attn.cpp attn.h ppnt.h main.h
# 	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
# 	-o $@ $<

clean: 
	rm -f $(TARGET)