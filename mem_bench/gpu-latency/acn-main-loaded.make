BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx950
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/lib/rocprofiler -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

PREFIX		:= acn
NAME 		:= main
SUFFIX		:= loaded
N 			:= 1

# for background bw load
# BPX			:= 
BPX			:= 256 # MI350X has 32 CUs per XCD
CCFLAGS   	+= -D BPX=$(BPX)

all: $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=0 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=1 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall: $(PREFIX)-$(NAME)-$(SUFFIX).hip 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall
