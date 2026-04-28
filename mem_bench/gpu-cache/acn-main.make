BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx950
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

NAME 		:= main
PREFIX		:= acn

all: $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-uniform 

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop0: $(PREFIX)-$(NAME).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=0 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1: $(PREFIX)-$(NAME).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=1 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-uniform: main.hip
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop1 
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-uniform