BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME := /opt/rocm-7.1.0
OPTS     := --amdgpu-target=gfx942
CC       := $(HIP_HOME)/bin/hipcc
CCFLAGS  :=
# Include rocprofiler/HSA headers; also add ../test so that ppnt.h etc. resolve
# their own relative includes (e.g. "main.h") even when compiled from this dir.
INCLUDES := -I$(HIP_HOME)/include/rocprofiler/ \
            -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS  := -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

DB_ALL_TARGET := $(BIN_DIR)/db_all

DB_LIST_SIZE := 64
DB_CANDIDATES_PER_QUERY := 64

.PHONY: all clean

all: $(DB_ALL_TARGET)

$(BIN_DIR):
	mkdir -p $@

$(DB_ALL_TARGET): db.cpp db.h \
                  ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DDB_LIST_SIZE=$(DB_LIST_SIZE) \
	-DDB_CANDIDATES_PER_QUERY=$(DB_CANDIDATES_PER_QUERY) \
	-o $@ $<

clean:
	rm -f $(DB_ALL_TARGET)
