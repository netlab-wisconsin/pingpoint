BASE_DIR := $(shell pwd)
HIP_HOME := /opt/rocm-7.1.0
TARGET ?= $(BASE_DIR)/bin/overhead_moe

CC := $(HIP_HOME)/bin/hipcc
CXXFLAGS := --offload-arch=gfx942 -O3 -Wall -Wextra
INCLUDES := -I$(BASE_DIR) -I$(BASE_DIR)/../moe -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS := -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl
DEPS := overhead_moe.cpp overhead_moe.h ../moe/main.h ../moe/k2.h ../../mem_bench/gpu-clock.cuh

.PHONY: all clean

all: $(TARGET)

$(BASE_DIR)/bin:
	mkdir -p $@

$(TARGET): $(DEPS) | $(BASE_DIR)/bin
	$(CC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ overhead_moe.cpp

clean:
	rm -f $(TARGET)
