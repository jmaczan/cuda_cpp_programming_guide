# cuda_cpp_programming_guide
Working through NVIDIA's CUDA C++ Programming Guide - version 12.8

## Notes

Thread -> Block -> Grid

Typically 1024 threads per block

- Block (Thread Block): A group of threads. Threads within the same block can:
    - Cooperate: By synchronizing their execution using __syncthreads(). This is a barrier; threads reaching it wait until all threads in the block reach it before any proceed.
    - Share Data: Using a fast, on-chip shared memory (__shared__). This is much faster than global device memory.

<<< GridSize, BlockSize, SharedMemoryBytes, Stream >>>

- GridSize: Specifies the dimensions of the grid (how many blocks). It can be an int (for 1D) or a dim3 type (for 1D, 2D, or 3D). dim3(Gx, Gy, Gz) means Gx * Gy * Gz total blocks.
- BlockSize: Specifies the dimensions of each block (how many threads per block). It can be an int or a dim3. dim3(Bx, By, Bz) means Bx * By * Bz threads per block. The total threads launched is GridSize * BlockSize.
- SharedMemoryBytes (Optional): Amount of dynamic shared memory to allocate per block (defaults to 0).
- Stream (Optional): Specifies the CUDA stream for asynchronous execution (defaults to the default stream 0).
