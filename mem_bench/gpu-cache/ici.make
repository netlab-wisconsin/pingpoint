BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

NAME 		:= cache
PREFIX		:= ici

all: $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-hop2 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-hop3 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-uniform 

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop0: ici-main.cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=0 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1: ici-main.cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=1 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop2: ici-main.cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=2 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop3: ici-main.cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=3 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-uniform: main.hip
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop1 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop2 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop3 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-uniform
