BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME := /opt/rocm-7.1.0
OPTS     := --amdgpu-target=gfx942
CC       := $(HIP_HOME)/bin/hipcc
CCFLAGS  :=
# Include rocprofiler/HSA headers; also add ../test so that ppnt.h etc. resolve
# their own relative includes (e.g. "main.h") even when compiled from this dir.
INCLUDES := -I$(HIP_HOME)/include/rocprofiler/ \
            -I$(HIP_HOME)/hsa/include/hsa \
            -I$(BASE_DIR)/../test
LDFLAGS  := -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

DB_ALL_TARGET := $(BIN_DIR)/db_all

.PHONY: all clean

all: $(DB_ALL_TARGET)

$(BIN_DIR):
	mkdir -p $@

$(DB_ALL_TARGET): db.cpp db.h \
                  ../test/ppnt.h ../test/main.h ../test/k1.h ../test/k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-o $@ $<

clean:
	rm -f $(DB_ALL_TARGET)
