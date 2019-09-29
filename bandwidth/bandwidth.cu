#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <cuda.h>

#include <numa.h>

float cuda_malloc_test( int size, bool up, int numa_node) {
    cudaEvent_t start, stop;
    cudaEventCreate( & start);
    cudaEventCreate( & stop);

    int * a = static_cast< int * >( ::numa_alloc_onnode( size * sizeof( int), numa_node) );
    int * dev_a;
    cudaMalloc( & dev_a, size * sizeof( * dev_a ) );

    cudaEventRecord( start, 0);
    for ( int i = 0; i < 100; ++i) {
        if ( up) {
            cudaMemcpy( dev_a, a, size * sizeof( * dev_a), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy( a, dev_a, size * sizeof( * dev_a), cudaMemcpyDeviceToHost);
        }
    }
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop);
    float elapsedTime;
    cudaEventElapsedTime( & elapsedTime, start, stop);

    numa_free( a, size * sizeof( int) );
    cudaFree( dev_a);
    cudaEventDestroy( start);
    cudaEventDestroy( stop);

    return elapsedTime;
}

int main( int argc, char * argv[]) {
    int numa_node = std::atoi( argv[1]);
    ::numa_run_on_node( numa_node);
    std::printf("NUMA node: %d\n", numa_node);
    for ( int dev = 0; dev < 2; ++dev) {
        cudaSetDevice( dev);

        constexpr int size = 32 * 1024 * 1024 * sizeof( int);
        constexpr float gb = static_cast< float >( 100) * size * sizeof( int)/(1024 * 1024 * 1024);

        int curr_dev = -1;
        cudaGetDevice( & curr_dev);
        std::printf("GPU: %d\n", curr_dev);

        float elapsedTime = cuda_malloc_test( size, true, numa_node);
        std::printf("host-to-device: %3.1f GB/s\n", gb / (elapsedTime/1000) );

        elapsedTime = cuda_malloc_test( size, false, numa_node);
        std::printf("device-to-host: %3.1f GB/s\n", gb / (elapsedTime/1000) );
    }

    return EXIT_SUCCESS;
}
