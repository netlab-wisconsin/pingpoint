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
P_WORKERS := 4
P_ACTIVE_XCD := 2
# Valid values: 0, 1, 2, 3
P_TARGET_CC := 0
P_ARRIVAL_QPS := 700000
BE_WORKERS_PER_XCD := 8
BE_BATCH_CHUNKS := 64
# Policy c: probability (0-100) a random BE batch lands on P's CC (25 = uniform)
BE_RANDOM_CC0_PCT := 50
# Policy d: percentage of BE batches randomly skipped
BE_THROTTLE_PCT := 75
P_MEASURE_LOAD_LATENCY := 0
BE_MEASURE_LOAD_LATENCY := 0
DB_USE_GLOBAL_BARRIER := 0 # set for profiling

DEBUG_DB := 1
DEBUG_LEVEL := 1

DB_A_TARGET := $(BIN_DIR)/db-a_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}
DB_B_TARGET := $(BIN_DIR)/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_$(BE_WORKERS_PER_XCD)_$(BE_BATCH_CHUNKS)
DB_C_TARGET := $(BIN_DIR)/db-c_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_$(BE_WORKERS_PER_XCD)_$(BE_BATCH_CHUNKS)_R_$(BE_RANDOM_CC0_PCT)
DB_D_TARGET := $(BIN_DIR)/db-d_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_$(BE_WORKERS_PER_XCD)_$(BE_BATCH_CHUNKS)_T_$(BE_THROTTLE_PCT)
DB_E_TARGET := $(BIN_DIR)/db-e_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_$(BE_WORKERS_PER_XCD)_$(BE_BATCH_CHUNKS)

DB_B_COMMON_DEFS := \
	-DDB_LIST_SIZE=$(DB_LIST_SIZE) \
	-DDB_CANDIDATES_PER_QUERY=$(DB_CANDIDATES_PER_QUERY) \
	-DP_WORKERS=$(P_WORKERS) \
	-DP_ACTIVE_XCD=$(P_ACTIVE_XCD) \
	-DP_TARGET_CC=$(P_TARGET_CC) \
	-DP_ARRIVAL_QPS=$(P_ARRIVAL_QPS) \
	-DBE_WORKERS_PER_XCD=$(BE_WORKERS_PER_XCD) \
	-DBE_BATCH_CHUNKS=$(BE_BATCH_CHUNKS) \
	-DP_MEASURE_LOAD_LATENCY=$(P_MEASURE_LOAD_LATENCY) \
	-DBE_MEASURE_LOAD_LATENCY=$(BE_MEASURE_LOAD_LATENCY) \
	-DDB_USE_GLOBAL_BARRIER=$(DB_USE_GLOBAL_BARRIER) \
	-DDEBUG_DB=$(DEBUG_DB) \
	-DDEBUG_LEVEL=$(DEBUG_LEVEL)

.PHONY: all policy-a policy-b policy-c policy-d policy-e clean

all: $(DB_A_TARGET) $(DB_B_TARGET) $(DB_C_TARGET) $(DB_D_TARGET) $(DB_E_TARGET)

policy-a: $(DB_A_TARGET)

policy-b: $(DB_B_TARGET)

policy-c: $(DB_C_TARGET)

policy-d: $(DB_D_TARGET)

policy-e: $(DB_E_TARGET)

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
	-DP_MEASURE_LOAD_LATENCY=$(P_MEASURE_LOAD_LATENCY) \
	-DDB_USE_GLOBAL_BARRIER=$(DB_USE_GLOBAL_BARRIER) \
	-DDEBUG_DB=$(DEBUG_DB) \
	-DDEBUG_LEVEL=$(DEBUG_LEVEL) \
	-o $@ $<

# Policy b: every BE batch targets P_TARGET_CC.
$(DB_B_TARGET): db-b.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	$(DB_B_COMMON_DEFS) \
	-o $@ $<

# Policy c: each BE batch targets a random CC; BE_RANDOM_CC0_PCT biases how
# often the destination collides with P's CC (25 = uniform, 100 = policy b).
$(DB_C_TARGET): db-c.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	$(DB_B_COMMON_DEFS) \
	-DBE_TARGET_POLICY=1 \
	-DBE_RANDOM_CC0_PCT=$(BE_RANDOM_CC0_PCT) \
	-o $@ $<

# Policy d: BE keeps targeting P's CC but randomly skips BE_THROTTLE_PCT% of
# batches (random throttling: removes work instead of moving it).
$(DB_D_TARGET): db-d.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	$(DB_B_COMMON_DEFS) \
	-DBE_THROTTLE_PCT=$(BE_THROTTLE_PCT) \
	-o $@ $<

# Policy e: PingPoint-guided migration — BE batches go to non-P CCs only.
$(DB_E_TARGET): db-e.cpp db.h \
                ppnt.h main.h k1.h k2.h | $(BIN_DIR)
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) \
	$(DB_B_COMMON_DEFS) \
	-DBE_TARGET_POLICY=2 \
	-o $@ $<

clean:
	rm -f $(DB_A_TARGET) $(DB_B_TARGET) $(DB_C_TARGET) $(DB_D_TARGET) $(DB_E_TARGET)
