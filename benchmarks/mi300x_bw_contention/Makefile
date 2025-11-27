BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_PATH := /opt/rocm-7.1.0
CC       := $(HIP_PATH)/bin/hipcc
HIPOPTS  := --amdgpu-target=gfx942
# HIPOPTS  += --save-temps

TPX_MAX := 16 # thread blocks per xcd
BENCHMARKS := 

BMK1 := bmk1
BENCHMARKS += $(addprefix $(BMK1)_tpx_,$(shell seq 1 $(TPX_MAX)))

BMK2 := unloaded_local
BENCHMARKS += $(addprefix $(BMK2)_tpx_,$(shell seq 1 $(TPX_MAX)))

all: $(BENCHMARKS)

$(BMK1)_tpx_%: $(BMK1).cpp
	$(CC) $(HIPOPTS) -DTPX=$* -o $(BIN_DIR)/$@ $<

$(BMK2)_tpx_%: $(BMK2).cpp
	$(CC) $(HIPOPTS) -DTPX=$* -o $(BIN_DIR)/$@ $<

clean:
	rm -f $(addprefix $(BIN_DIR)/,$(BENCHMARKS))
