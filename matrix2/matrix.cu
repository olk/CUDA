#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

constexpr int TILE_DIM = 32;

__global__
void multiplication( int * a, int * b, int * c, int dim) {
    // create shared memory
    __shared__ int m[TILE_DIM][TILE_DIM];
    __shared__ int n[TILE_DIM][TILE_DIM];
    // indices will be stored in registers (automatic variables)
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    // compute the columns and row (TILE_DIM == blockDim.x/y)
    int column = tx + bx * TILE_DIM;
    int row = ty + by * TILE_DIM;
    int value = 0;
    int limit = std::ceil( dim/static_cast< float >( TILE_DIM) );
    // loop over the m and n tiles
    // strip-mining: break a long-running loop into phases
    //               each phase consists of an inner loop
    //               that executes a number of consecutive
    //               steps of the original loop
    for ( int phase = 0; phase < limit; ++phase) {
        // load tiles into shared memory
        m[ty][tx] = a[row * dim + phase * TILE_DIM + tx];
        n[ty][tx] = b[(phase * TILE_DIM + ty) * dim + column];
        // wait till all threads in the block have finished loading
        // the tiles into shared memory
        __syncthreads();
        // compute the dot product 
        for ( int k = 0; k < TILE_DIM; ++k) {
            value += m[ty][k] * n[k][tx];
        }
        // wait till all threads in the block have finished
        // computing the dot product
        __syncthreads();
    }
    c[row * dim + column] = value;
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
