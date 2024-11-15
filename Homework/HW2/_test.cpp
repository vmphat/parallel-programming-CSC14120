#include <stdio.h>
#include <stdlib.h>

void sumArrayVer1(int *in, int blockIdx, int blockDim, int threadIdx, int n, int stride)
{
    // Number of data elements of all previous blocks
    int numElemsBeforeBlk = blockIdx * blockDim * 2;
    // Index of the first element of the thread
    int thrStart = numElemsBeforeBlk + threadIdx * 2;
    // If the index is out of range, return
    if (thrStart >= n || threadIdx % stride != 0) {
        return;
    }

    // Print Information
    printf("  - threadIdx: %d, dataIdx:  %d", threadIdx, thrStart);
    if (thrStart + stride < n) {
        printf(" <opr> %d", thrStart + stride);
    } 
    printf("\n");
}

void sumArrayVer3(int *in, int blockIdx, int blockDim, int threadIdx, int n, int stride)
{
    // Number of data elements of all previous blocks
    int numElemsBeforeBlk = blockIdx * blockDim * 2;
    // Index of the first element of the thread
    int thrStart = numElemsBeforeBlk + threadIdx;
    // If the index is out of range, return
    if (thrStart >= n || threadIdx >= stride) {
        return;
    }

    // Print Information
    printf("  - threadIdx: %d, dataIdx:  %d", threadIdx, thrStart);
    if (thrStart + stride < n) {
        printf(" <opr> %d", thrStart + stride);
    } 
    printf("\n");
}


int main(int argc, char **argv) {
    int N = 16;
    int blockDim = 8;
    if (argc == 3) {
        N = atoi(argv[1]);
        blockDim = atoi(argv[2]);
    }
    // Input array
    int in[N] = {};
    for (int i = 0; i < N; i++) { in[i] = i; }
    // Number of elements each block will process
    int numElemsPerBlk = blockDim * 2;
    // Number of blocks required
    int numBlks = (N - 1) / numElemsPerBlk + 1;

    // // Print Information
    // printf("N: %d, blockDim: %d => numBlks: %d\n", N, blockDim, numBlks);

    // // Loop over all blocks
    // for (int blockIdx = 0; blockIdx < numBlks; blockIdx++) {
    //     printf("blockIdx: %d\n", blockIdx);

    //     // Loop over all strides
    //     // for (int stride = 1; stride < blockDim * 2; stride *= 2) {
    //     for (int stride = blockDim / 2; stride > 0; stride /= 2) {
    //         printf(" * stride: %d\n", stride);

    //         // Loop over all threads
    //         for (int threadIdx = 0; threadIdx < blockDim; threadIdx++) {
    //             // sumArrayVer1(in, blockIdx, blockDim, threadIdx, N, stride);
    //             sumArrayVer3(in, blockIdx, blockDim, threadIdx, N, stride);
    //         }
    //     }
    // }
    
    int m = (1 << 10);
    int n = (1 << 9);
    int k = (1 << 10);

    size_t size_A = m * n;
    size_t size_B = n * k;
    size_t size_C = m * k;

    printf("Size A + B: %ld\n", size_A + size_B);
    printf("Size C: %ld\n", size_C);
    return 0;
}