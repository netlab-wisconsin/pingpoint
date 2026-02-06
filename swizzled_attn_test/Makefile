BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

SUFFIX := run

TARGET := $(BIN_DIR)/swizzled_attn_$(SUFFIX)

all: $(TARGET)

$(TARGET): main.cpp swizzled_attn.h ppnt.h main.h k1.h k2.h
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

clean: 
	rm -f $(BIN_DIR)/swizzled_attn_*