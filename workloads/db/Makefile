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

DB_LIST_SIZE := 64
DB_CANDIDATES_PER_QUERY := 1
P_WORKERS := 8
P_ACTIVE_XCD := 2
# Valid values: 0, 1, 2, 3
P_TARGET_CC := 0
P_ARRIVAL_QPS := 500000
BE_WORKERS_PER_XCD := 8
BE_BATCH_CHUNKS := 64

DEBUG_DB := 0
DEBUG_LEVEL := 0

DB_A_TARGET := $(BIN_DIR)/db-a_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}
DB_B_TARGET := $(BIN_DIR)/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_$(BE_WORKERS_PER_XCD)_$(BE_BATCH_CHUNKS)

.PHONY: all policy-a policy-b clean

all: $(DB_A_TARGET) $(DB_B_TARGET)

policy-a: $(DB_A_TARGET)

policy-b: $(DB_B_TARGET)

$(BIN_DIR):
	mkdir -p $@

$(DB_A_TARGET): db-a.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DDB_LIST_SIZE=$(DB_LIST_SIZE) \
	-DDB_CANDIDATES_PER_QUERY=$(DB_CANDIDATES_PER_QUERY) \
	-DP_WORKERS=$(P_WORKERS) \
	-DP_ACTIVE_XCD=$(P_ACTIVE_XCD) \
	-DP_TARGET_CC=$(P_TARGET_CC) \
	-DP_ARRIVAL_QPS=$(P_ARRIVAL_QPS) \
	-DDEBUG_DB=$(DEBUG_DB) \
	-DDEBUG_LEVEL=$(DEBUG_LEVEL) \
	-o $@ $<

$(DB_B_TARGET): db-b.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	-DDB_LIST_SIZE=$(DB_LIST_SIZE) \
	-DDB_CANDIDATES_PER_QUERY=$(DB_CANDIDATES_PER_QUERY) \
	-DP_WORKERS=$(P_WORKERS) \
	-DP_ACTIVE_XCD=$(P_ACTIVE_XCD) \
	-DP_TARGET_CC=$(P_TARGET_CC) \
	-DP_ARRIVAL_QPS=$(P_ARRIVAL_QPS) \
	-DBE_WORKERS_PER_XCD=$(BE_WORKERS_PER_XCD) \
	-DBE_BATCH_CHUNKS=$(BE_BATCH_CHUNKS) \
	-DDEBUG_DB=$(DEBUG_DB) \
	-DDEBUG_LEVEL=$(DEBUG_LEVEL) \
	-o $@ $<

clean:
	rm -f $(DB_A_TARGET) $(DB_B_TARGET)
