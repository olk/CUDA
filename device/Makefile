CUDA_PATH := /opt/cuda

NVCC := $(CUDA_PATH)/bin/nvcc

CPPFLAGS := -O2 -std=c++11
LDFLAGS := -g -L/usr/local/lib
INCLUDES := -I/usr/local/include -I$(CUDA_PATH)/include
LIBRARIES := 

all: build

build: device

device.o:device.cu
	$(NVCC) $(INCLUDES) $(CPPFLAGS) -o $@ -c $<

device: device.o
	$(NVCC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f device device.o

clobber: clean
