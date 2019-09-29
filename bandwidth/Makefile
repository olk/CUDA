CUDA_PATH := /opt/cuda

NVCC := $(CUDA_PATH)/bin/nvcc

CPPFLAGS := -O2 -std=c++11
LDFLAGS := -g -L/usr/local/lib
INCLUDES := -I/usr/local/include -I$(CUDA_PATH)/include
LIBRARIES := -lnuma

all: build

build: bandwidth

bandwidth.o:bandwidth.cu
	$(NVCC) $(INCLUDES) $(CPPFLAGS) -o $@ -c $<

bandwidth: bandwidth.o
	$(NVCC) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f bandwidth bandwidth.o

clobber: clean
