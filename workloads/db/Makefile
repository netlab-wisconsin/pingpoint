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
DB_BE_TARGET  := $(BIN_DIR)/db_be

DB_LIST_SIZE := 64
DB_CANDIDATES_PER_QUERY := 1
P_WORKERS := 8
P_ACTIVE_XCD := 2
P_TARGET_CC := 2
P_ARRIVAL_QPS := 500000

DEBUG_DB := 0
DEBUG_LEVEL := 0

.PHONY: all clean

all: $(DB_ALL_TARGET) $(DB_BE_TARGET)

$(BIN_DIR):
	mkdir -p $@

$(DB_ALL_TARGET): db.cpp db.h \
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

# $(DB_BE_TARGET): db_be.cpp db.h \
#                  ppnt.h main.h k1.h k2.h | $(BIN_DIR)
# 	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
# 	-DDB_LIST_SIZE=$(DB_LIST_SIZE) \
# 	-DDB_CANDIDATES_PER_QUERY=$(DB_CANDIDATES_PER_QUERY) \
# 	-DP_WORKERS=$(P_WORKERS) \
# 	-DP_ACTIVE_XCD=$(P_ACTIVE_XCD) \
# 	-DP_TARGET_CC=$(P_TARGET_CC) \
# 	-DP_ARRIVAL_QPS=$(P_ARRIVAL_QPS) \
# 	-DDEBUG_DB=$(DEBUG_DB) \
# 	-DDEBUG_LEVEL=$(DEBUG_LEVEL) \
# 	-o $@ $<

clean:
	rm -f $(DB_ALL_TARGET) $(DB_BE_TARGET)
