BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_PATH := /opt/rocm-7.1.0
CC       := $(HIP_PATH)/bin/hipcc
HIPOPTS  := --amdgpu-target=gfx942
# HIPOPTS  += --save-temps

BMK1_TB := bmk1_tb

TPX_MAX := 16 # thread blocks per xcd

BENCHMARKS := 
BENCHMARKS += $(addprefix $(BMK1_TB)_tpx_,$(shell seq 1 $(TPX_MAX)))

all: $(BENCHMARKS)

$(BMK1_TB)_tpx_%: $(BMK1_TB).cpp
	$(CC) $(HIPOPTS) -DTPX=$* -o $(BIN_DIR)/$@ $<

clean:
	rm -f $(addprefix $(BIN_DIR)/,$(BENCHMARKS))
