BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

HIPOPTS = --offload-arch=gfx950
HIP_PATH = /opt/rocm-7.1.0
CC := $(HIP_PATH)/bin/hipcc

BENCHMARKS := bmk1 bmk1_shuffle bmk1_ntbx bmk1_1tbx bmk2 bmk3

DEBUG ?= 0
LOG_CYCLE ?= 1
all: $(BENCHMARKS)

%: %.cpp
	$(CC) $(HIPOPTS) -DDEBUG=${DEBUG} -DLOG_CYCLE=${LOG_CYCLE} -o $(BIN_DIR)/$@ $<

clean:
	rm -f $(foreach bmk,$(BENCHMARKS),$(BIN_DIR)/$(bmk))

