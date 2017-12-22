#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

__global__
void multiplication( int * a, int * b, int * c, int dim) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if ( (row < dim) && (column < dim) ) {
        int value = 0;
        for ( int k = 0; k < dim; ++k) {
            value += a[row * dim + k] * b[k * dim + column];
        }
        c[row * dim + column] = value;
    }
}

void compute_on_device( int dim, int * host_a, int * host_b, int * host_c) {
    constexpr int tile_dim = 32;
    // allocate device memory
    int * device_a, * device_b, * device_c;
    cudaMalloc( & device_a, dim * dim * sizeof( int) );
    cudaMalloc( & device_b, dim * dim * sizeof( int) );
    cudaMalloc( & device_c, dim * dim * sizeof( int) );
    // copy input matrices from host to device memory
    cudaMemcpy( device_a, host_a, dim * dim * sizeof( int), cudaMemcpyHostToDevice);
    cudaMemcpy( device_b, host_b, dim * dim * sizeof( int), cudaMemcpyHostToDevice);
    dim3 block_dim{ tile_dim, tile_dim };
    dim3 grid_dim{ static_cast< unsigned int >( std::ceil( dim/static_cast< float >( block_dim.x) ) ),
                   static_cast< unsigned int >( std::ceil( dim/static_cast< float >( block_dim.y) ) ) };
    auto start = std::chrono::high_resolution_clock::now();
    multiplication<<< grid_dim, block_dim >>>( device_a, device_b, device_c, dim);
    cudaDeviceSynchronize();
    auto duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "device: " << std::chrono::duration_cast< std::chrono::microseconds >( duration).count() << " ms\n";
    cudaMemcpy( host_c, device_c, dim * dim * sizeof( int), cudaMemcpyDeviceToHost);
    cudaFree( device_a);
    cudaFree( device_b);
    cudaFree( device_c);
}

void compute_on_host( int dim, int * a, int * b, int * c) {
    auto start = std::chrono::high_resolution_clock::now();
    for ( int row = 0; row < dim; ++row) {
        for ( int column = 0; column < dim; ++column) {
            for ( int k = 0; k < dim; ++k) {
                c[row * dim + column] += a[row * dim + k] * b[k * dim + column];
            }
        }
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "host: " << std::chrono::duration_cast< std::chrono::microseconds >( duration).count() << " ms\n";
}

bool equal( int dim, int * host, int * device) {
    for ( int row = 0; row < dim; ++row) {
        for ( int column = 0; column < dim; ++column) {
            if ( host[row * dim + column] != device[row * dim + column]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    constexpr int dim = 1024;
    // allocate host memory
    int * host_a = static_cast< int * >( std::malloc( dim * dim * sizeof( int) ) );
    int * host_b = static_cast< int * >( std::malloc( dim * dim * sizeof( int) ) );
    // initialize input matrices
    std::minstd_rand generator;
    std::uniform_int_distribution<> distribution{ 0, 255 };
    for ( unsigned int i = 0; i < dim*dim; ++i) {
        host_a[i] = distribution( generator); 
        host_b[i] = host_a[i];
    }
    // multiplication on host
    int * host_c = static_cast< int * >( std::malloc( dim * dim * sizeof( int) ) );
    compute_on_host( dim, host_a, host_b, host_c);
    // multiplication on device
    int * device_c = static_cast< int * >( std::malloc( dim * dim * sizeof( int) ) );
    compute_on_device( dim, host_a, host_b, device_c);
    if ( ! equal( dim, host_c, device_c) ) {
        std::cout << "matrices are not equal" << std::endl;
    }
    std::free( host_a);
    std::free( host_b);
    std::free( host_c);
    std::free( device_c);
    return EXIT_SUCCESS;
}
