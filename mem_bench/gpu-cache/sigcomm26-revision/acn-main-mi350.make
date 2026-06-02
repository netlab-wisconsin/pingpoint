BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/../bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx950
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

NAME 		:= main
PREFIX		:= acn
ARCH		:= mi350
SUFFIX		:= L1 # Set accordingly!

all: $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0-$(ARCH)-$(SUFFIX) \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1-$(ARCH)-$(SUFFIX)

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop0-$(ARCH)-$(SUFFIX): $(PREFIX)-$(NAME)-$(ARCH).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=0 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-hop1-$(ARCH)-$(SUFFIX): $(PREFIX)-$(NAME)-$(ARCH).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DInterCCHop=1 -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop0-$(ARCH)-$(SUFFIX)
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-hop1-$(ARCH)-$(SUFFIX)