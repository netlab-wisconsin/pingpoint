BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

NAME 		:= stream
PREFIX		:= ici

all: $(BIN_DIR)/$(PREFIX)-hip-$(NAME)

$(BIN_DIR)/$(PREFIX)-hip-$(NAME): ici-main.cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-hip-$(NAME)


