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

int main() {
    std::vector< cudaDeviceProp > devs = get_cuda_device();
    for ( cudaDeviceProp dev: devs) {
        int block_8x8 = 8*8;
        int block_16x16 = 16*16;
        int block_32x32 = 32*32;

        auto sm_device = dev.multiProcessorCount;
        auto warp_size = dev.warpSize;
        auto threads_sm = dev.maxThreadsPerMultiProcessor;
        auto threads_block = dev.maxThreadsPerBlock;
        auto registers_sm = dev.regsPerMultiprocessor;
        auto shared_mem_sm = dev.sharedMemPerMultiprocessor;
        auto shared_mem_block = dev.sharedMemPerBlock;

        std::printf("SMs pro device: %d\n", sm_device);
        std::printf("max. threads pro SM: %d\n", threads_sm);
        std::printf("registers pro SM: %d\n", registers_sm);
        std::printf("shared memory pro SM: %d kB\n", shared_mem_sm / 1024);
        std::printf("max. threads pro block: %d\n", threads_block);
        std::printf("max. shared memory pro block: %d kB\n", shared_mem_block / 1024);
        std::printf("threads pro warp: %d\n", warp_size);
        std::printf("\n");

        auto fn = [threads_sm,warp_size,registers_sm,shared_mem_sm](int threads_block){
                int blocks_sm = threads_sm / threads_block;
                int warps_block = threads_block / warp_size;
                int shared_mem_block = shared_mem_sm / blocks_sm;

                std::printf("threads pro block: %d\n", threads_block);
                std::printf("blocks pro SM: %d\n", blocks_sm);
                std::printf("warps pro block: %d\n", warps_block);
                std::printf("shared memory pro block: %d kB\n", shared_mem_block / 1024);
                std::printf("\n");
        };

        fn( block_8x8);
        fn( block_16x16);
        fn( block_32x32);
    }
    return EXIT_SUCCESS;
}
