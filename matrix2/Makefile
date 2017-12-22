CUDA_PATH := /opt/cuda

NVCC := $(CUDA_PATH)/bin/nvcc

CPPFLAGS := -O2 -std=c++11
LDFLAGS := -g -L/usr/local/lib
INCLUDES := -I/usr/local/include -I$(CUDA_PATH)/include
LIBRARIES := 

all: build

build: matrix

matrix.o:matrix.cu
	$(NVCC) $(INCLUDES) $(CPPFLAGS) -o $@ -c $<

matrix: matrix.o
	$(NVCC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f matrix matrix.o

clobber: clean
