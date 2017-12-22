# Location of the CUDA Toolkit
CUDA_PATH ?= /opt/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
TARGET_SIZE := 64

# host compiler
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER) 

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

EXTRA_NVCCFLAGS = -std=c++11

ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include
LIBRARIES :=

################################################################################

# Gencode arguments
# Supported on CUDA 8 and later
#  SM61 – GTX 1080, 1070, 1060, Titan Xp, Tesla P40, Tesla P4
#  SM62 – Probably Drive-PX2
# Supported on CUDA 9 and later
#  SM70 – Tesla V100
SM ?= 61

# Generate PTX code from the highest SM architecture in $(SM) to guarantee forward-compatibility
GENCODE_FLAGS += -gencode arch=compute_$(SM),code=compute_$(SM)

################################################################################

# Target rules
all: build

build: device_properties

device_properties.o:device_properties.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

device_properties: device_properties.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./device_properties

clean:
	rm -f device_properties device_properties.o

clobber: clean
