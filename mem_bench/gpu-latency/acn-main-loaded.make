BASE_DIR := $(shell pwd)
BIN_DIR  := $(BASE_DIR)/bin

HIP_HOME 	:= /opt/rocm-7.1.0
OPTS 		:= --amdgpu-target=gfx942
CC 			:= $(HIP_HOME)/bin/hipcc
CCFLAGS 	:= 
INCLUDES 	:= -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa
LDFLAGS 	:= -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl

PREFIX		:= acn
NAME 		:= main
SUFFIX		:= loaded
N 			:= 1

# for background bw load
BPX			:= 304
CCFLAGS   	+= -D BPX=$(BPX)

all: $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc2 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc3 \
	$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=0 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=1 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc2: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=2 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc3: $(PREFIX)-$(NAME)-$(SUFFIX).cpp 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -DPINNED_CC=3 -o $@ $<

$(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall: $(PREFIX)-$(NAME)-$(SUFFIX).hip 
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc0
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc1
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc2
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-cc3
	rm -f $(BIN_DIR)/$(PREFIX)-$(NAME)-$(SUFFIX)-ccall