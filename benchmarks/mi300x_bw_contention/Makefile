BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_PATH := /opt/rocm-7.1.0
CC       := $(HIP_PATH)/bin/hipcc
HIPOPTS  := --amdgpu-target=gfx942
# HIPOPTS  += --save-temps

BMK1 := bmk1

TPX_MAX := 16 # thread blocks per xcd

BENCHMARKS := 
BENCHMARKS += $(addprefix $(BMK1)_tpx_,$(shell seq 1 $(TPX_MAX)))

all: $(BENCHMARKS)

$(BMK1)_tpx_%: $(BMK1).cpp
	$(CC) $(HIPOPTS) -DTPX=$* -o $(BIN_DIR)/$@ $<

clean:
	rm -f $(addprefix $(BIN_DIR)/,$(BENCHMARKS))
