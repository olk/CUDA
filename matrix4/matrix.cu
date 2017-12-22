#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

__global__
void multiplication( int * a, int * b, int * c,
                     int a_rows, int a_columns,
                     int b_rows, int b_columns,
                     int c_rows, int c_columns,
                     int tile_dim) {
    // create shared memory
    extern __shared__ int shared[];
    int * m = shared;
    int * n = & shared[tile_dim * tile_dim];
    // indices will be stored in registers (automatic variables)
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    // compute the columns and row (tile_dim == blockDim.x/y)
    int column = tx + bx * tile_dim;
    int row = ty + by * tile_dim;
    int value = 0;
    int limit = std::ceil( a_columns/static_cast< float >( tile_dim) );
    // loop over the m and n tiles
    // strip-mining: break a long-running loop into phases
    //               each phase consists of an inner loop
    //               that executes a number of consecutive
    //               steps of the original loop
    for ( int phase = 0; phase < limit; ++phase) {
        // load tiles into shared memory
        // boundary condition check for matrix m
        if ( row < a_rows && (phase * tile_dim + tx) < a_columns) {
            m[ty * tile_dim + tx] = a[row * a_columns + phase * tile_dim + tx];
        } else {
            m[ty * tile_dim + tx] = 0;
        }
        // boundary condition check for matrix n
        if ( (phase * tile_dim + ty) < b_rows && column < b_columns) {
            n[ty * tile_dim + tx] = b[(phase * tile_dim + ty) * b_columns + column];
        } else {
            n[ty * tile_dim + tx] = 0;
        }
        // wait till all threads in the block have finished loading
        // the tiles into shared memory
        __syncthreads();
        // compute the dot product 
        for ( int k = 0; k < tile_dim; ++k) {
            value += m[ty * tile_dim + k] * n[k * tile_dim + tx];
        }
        // wait till all threads in the block have finished
        // computing the dot product
        __syncthreads();
    }
    // boundary condition check for matrix c
    if ( row < c_rows && column < c_columns) {
        c[row * c_columns + column] = value;
    }
}

void compute_on_device( int a_rows, int a_columns, int * host_a,
                               int b_rows, int b_columns, int * host_b,
                               int c_rows, int c_columns, int * host_c) {
    constexpr int tile_dim = 32;
    // allocate device memory for one dimensional representation of the matrix
    int * device_a, * device_b, * device_c;
    cudaMalloc( & device_a, a_rows * a_columns * sizeof( int) );
    cudaMalloc( & device_b, b_rows * b_columns * sizeof( int) );
    cudaMalloc( & device_c, c_rows * c_columns * sizeof( int) );
    // copy input matrices from host to device memory
    cudaMemcpy( device_a, host_a, a_rows * a_columns * sizeof( int), cudaMemcpyHostToDevice);
    cudaMemcpy( device_b, host_b, b_rows * b_columns * sizeof( int), cudaMemcpyHostToDevice);
    // block dimension should be chosen equal to tile dimension
    // matrix dimensions are multiples of the tile dimension
    dim3 block_dim{ tile_dim, tile_dim };
    dim3 grid_dim{ static_cast< unsigned int >( std::ceil( c_columns/static_cast< float >( block_dim.x) ) ),
                   static_cast< unsigned int >( std::ceil( c_rows/static_cast< float >( block_dim.y) ) ) };
    std::cout << "block_dim.x == " << block_dim.x << ", block_dim.y == " << block_dim.y << std::endl;
    std::cout << "grid_dim.x == " << grid_dim.x << ", grid_dim.y == " << grid_dim.y << std::endl;
    std::size_t sm_size = 2 * tile_dim * tile_dim * sizeof( int);
    auto start = std::chrono::high_resolution_clock::now();
    multiplication<<< grid_dim, block_dim, sm_size >>>( device_a, device_b, device_c, a_rows, a_columns, b_rows, b_columns, c_rows, c_columns, tile_dim);
    cudaDeviceSynchronize();
    auto duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "device: " << std::chrono::duration_cast< std::chrono::microseconds >( duration).count() << " ms\n";
    cudaMemcpy( host_c, device_c, c_rows * c_columns * sizeof( int), cudaMemcpyDeviceToHost);
    cudaFree( device_a);
    cudaFree( device_b);
    cudaFree( device_c);
}

void compute_on_host( int a_rows, int a_columns, int * a,
                      int b_rows, int b_columns, int * b,
                      int c_rows, int c_columns, int * c) {
    assert( b_rows == a_columns);
    assert( c_rows == a_rows);
    assert( c_columns == b_columns);
    auto start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < a_rows; ++i) {
        for ( int j = 0; j < b_columns; ++j) {
            c[i * c_columns + j] = 0;
            for ( int k = 0; k < b_rows; ++k) {
                c[i * c_columns + j] += a[i * a_columns + k] * b[k * b_columns + j];
            }
        }
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "host: " << std::chrono::duration_cast< std::chrono::microseconds >( duration).count() << " ms\n";
}

bool equal( int rows, int columns, int * host, int * device) {
    for ( int i = 0; i < rows * columns; ++i) {
        if ( host[i] != device[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int a_rows = 300;
    constexpr int a_columns = 1000;
    constexpr int b_rows = 1000;
    constexpr int b_columns = 500;
    constexpr int c_rows = a_rows;
    constexpr int c_columns = b_columns;
    // allocate host memory for one dimensional representation of the matrix
    int * host_a, * host_b;
    host_a = static_cast< int * >( std::malloc( a_rows * a_columns * sizeof( int) ) );
    host_b = static_cast< int * >( std::malloc( b_rows * b_columns * sizeof( int) ) );
    // initialize input matrices
    std::minstd_rand generator;
    std::uniform_int_distribution<> distribution{ 0, 255 };
    for ( unsigned int i = 0; i < a_rows * a_columns; ++i) {
        host_a[i] = distribution( generator); 
    }
    for ( unsigned int i = 0; i < b_rows * b_columns; ++i) {
        host_b[i] = distribution( generator); 
    }
    // multiplication on host
    int * host_c = static_cast< int * >( std::malloc( c_rows * c_columns * sizeof( int) ) );
    compute_on_host( a_rows, a_columns, host_a,
                     b_rows, b_columns, host_b,
                     c_rows, c_columns, host_c);
    // multiplication on device
    int * device_c = static_cast< int * >( std::malloc( c_rows * c_columns * sizeof( int) ) );
    compute_on_device( a_rows, a_columns, host_a,
                       b_rows, b_columns, host_b,
                       c_rows, c_columns, device_c);
    if ( ! equal( c_rows, c_columns, host_c, device_c) ) {
        std::cout << "matrices are not equal" << std::endl;
    }
    std::free( host_a);
    std::free( host_b);
    std::free( host_c);
    std::free( device_c);
    return EXIT_SUCCESS;
}
