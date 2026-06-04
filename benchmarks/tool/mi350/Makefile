BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

HIPOPTS = --amdgpu-target=gfx950
HIP_PATH = /opt/rocm-7.1.0
CC := $(HIP_PATH)/bin/hipcc

BENCHMARKS := bmk1 bmk1_shuffle bmk1_ntbx bmk1_1tbx bmk2 bmk3

all: $(BENCHMARKS)

%: %.cpp
	$(CC) $(HIPOPTS) -o $(BIN_DIR)/$@ $<

clean:
	rm -f $(foreach bmk,$(BENCHMARKS),$(BIN_DIR)/$(bmk))

