BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME ?= /opt/rocm-7.1.0
CC       := $(HIP_HOME)/bin/hipcc
OPTS     := --offload-arch=gfx942
INCLUDES := -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS  := -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

TARGET := $(BIN_DIR)/db_query_corun

.PHONY: all clean

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $@

$(TARGET): main.cpp db_kernels.h | $(BIN_DIR)
	$(CC) $(OPTS) $(INCLUDES) $(LDFLAGS) -o $@ main.cpp

clean:
	rm -f $(TARGET)
