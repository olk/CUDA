#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector< cudaDeviceProp > get_cuda_device() {
    std::vector< cudaDeviceProp > devices;
    int count = -1;
    cudaGetDeviceCount( & count);
    for ( int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( & prop, i);
        devices.push_back( prop);
    }
    return devices;
}

const char * compute_mode( cudaComputeMode m) {
    switch ( m) {
        case cudaComputeModeDefault:
            return "default";
        case cudaComputeModeExclusive:
            return "exclusive-thread";
        case cudaComputeModeProhibited:
            return "prohibited";
        case cudaComputeModeExclusiveProcess:
            return "exclusive-process";
        default:
            return "unknown";
    }
}

int main() {
    std::vector< cudaDeviceProp > devs = get_cuda_device();
    for ( cudaDeviceProp dev: devs) {
        std::printf("*** general device informations ***\n");
        std::printf("name: %s\n", dev.name);
        std::printf("compute capability: %d.%d\n", dev.major, dev.minor);
        std::printf("integrated: %s\n", dev.integrated ? "true" : "false");
        std::printf("multi-GPU board: %s\n", dev.isMultiGpuBoard ? "true" : "false");
        std::printf("compute mode: %s\n", compute_mode( static_cast< cudaComputeMode >( dev.computeMode) ) );
        std::printf("clock rate: %d MHz\n", dev.clockRate / 1000);
        std::printf("async engine count: %d\n", dev.asyncEngineCount);
        std::printf("kernel execution timeout: %s\n", dev.kernelExecTimeoutEnabled ? "enabled" : "disabled");
        std::printf("stream priorities: %s\n", dev.streamPrioritiesSupported ? "supported" : "not supported");
        std::printf("native atomic operations between device and host: %s\n", dev.hostNativeAtomicSupported ? "supported" : "not supported");
        std::printf("Tesla device with TCC driver: %s\n", dev.tccDriver ? "true" : "false");
        std::printf("ratio of single and double precision performance: %d\n", dev.singleToDoublePrecisionPerfRatio);
        std::printf("PCI bus ID: %d\n", dev.pciBusID);
        std::printf("PCI device ID: %d\n", dev.pciDeviceID);
        std::printf("PCI domain ID: %d\n", dev.pciDomainID);

        std::printf("\n*** memory informations ***\n");
        std::printf("total global memory: %ld MB\n", dev.totalGlobalMem / (1024*1024) );
        std::printf("total constant memory: %ld kB\n", dev.totalConstMem / (1024) );
        std::printf("global L1 cache: %s\n", dev.globalL1CacheSupported ? "supported" : "not supported");
        std::printf("local L1 cache: %s\n", dev.localL1CacheSupported ? "supported" : "not supported");
        std::printf("L2 cache size: %ld MB\n", dev.l2CacheSize / (1024*1024) );
        std::printf("memory bus width: %ld bit\n", dev.memoryBusWidth);
        std::printf("memory clock rate: %ld MHz\n", dev.memoryClockRate / 1000);
        std::printf("managed memory: %s\n", dev.managedMemory ? "supported" : "not supported");
        std::printf("unified addressing: %s\n", dev.unifiedAddressing ? "supported" : "not supported");
        std::printf("ECC:: %s\n", dev.ECCEnabled ? "enabled" : "disabled");
        std::printf("max memory pitch: %ld MB\n", dev.memPitch / (1024*1024) );
        std::printf("texture alignment: %ld\n", dev.textureAlignment);
        std::printf("texture pitch alignment: %ld\n", dev.texturePitchAlignment);
        std::printf("can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: %s\n", dev.canMapHostMemory ? "true" : "false");
        std::printf("access pageable memory concurrently without cudeHostRegister: %s\n", dev.pageableMemoryAccess ? "supported" : "not supported");

        std::printf("\n*** multiprocessing informations ***\n");
        std::printf("multiprocessor count: %d\n", dev.multiProcessorCount);
        std::printf("registers per block: %d\n", dev.regsPerBlock);
        std::printf("threads in warp: %d\n", dev.warpSize);
        std::printf("max threads per multiprocessor: %d\n", dev.maxThreadsPerMultiProcessor);
        std::printf("max threads per block: %d\n", dev.maxThreadsPerBlock);
        std::printf("max thread dimensions: {%d, %d, %d}\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
        std::printf("max grid dimensions: {%d, %d, %d}\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
        std::printf("concurrent kernels: %s\n", dev.concurrentKernels ? "supported" : "not supported");
        std::printf("registers per multiprocessor: %d\n", dev.regsPerMultiprocessor);
        std::printf("shared memory per multiprocessor: %d kB\n", dev.sharedMemPerMultiprocessor / 1024);
        std::printf("shared memory per block: %ld kB\n", dev.sharedMemPerBlock / 1024);
        std::printf("access managed memory concurrently with CPU: %s\n", dev.concurrentManagedAccess ? "supported" : "not supported");
        std::printf("\n\n");
    }
    return EXIT_SUCCESS;
}
