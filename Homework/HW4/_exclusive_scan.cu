// Exclusive scan of an array
#include <iostream>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start); 
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void exclusiveScanByHost(int *in, int *out, int n)
{
    printf("\nExclusive scan by host\n");
    GpuTimer timer; 
    timer.Start();
    out[0] = 0;
    for (int i = 1; i < n; ++i)
    {
        out[i] = out[i - 1] + in[i - 1];
    }
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}
void checkCorrectness(int *out1, int *out2, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (out1[i] != out2[i])
        {
            printf("\nMismatch at %d: %d != %d\n", i, out1[i], out2[i]);
            return;
        }
    }
    printf("\nCorrectness\n");
}


// Counter to get the dynamic block index for all blocks
__device__ int blockCounter1 = 0;
// Counter to get the sum of the previous block
__device__ int blockCounter2 = 0;
// Single-pass exclusive scan by device using Brent-Kung algorithm
__global__ void BrentKungExclusiveScanKernel(int *in, int *out, int n, int *blkSums)
{
    // ********** Step 0: Get in-order block index **********
    __shared__ int s_bid; // Shared dynamic block index
    if (threadIdx.x == 0)
    {
        /**
        * The global counter stores the dynamic block index 
        * of the next block that is scheduled
        */
        s_bid = atomicAdd(&blockCounter1, 1);
    }
    __syncthreads(); // Wait for leader thread to write s_bid
    // Load dynamic block index to register of each thread in the block
    int bid = s_bid;
    
    // ********** Step 1: Perform local scan in each block **********
    // 1.1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[]; // Size of s_data is 2 * blockDim.x
    int i1 = bid * blockDim.x * 2 + threadIdx.x;
    int i2 = i1 + blockDim.x;
    // Shift all data to the right by 1
    if (i1 < n && i1 > 0) // 0th-element in 0th-block is 0
    {
        s_data[threadIdx.x] = in[i1 - 1];
    }
    else
    {
        s_data[threadIdx.x] = 0;
    }
    if (i2 < n)
    {
        s_data[threadIdx.x + blockDim.x] = in[i2 - 1];
    }

    // 1.2. Each block will scan with data in SMEM
    // 1.2.1. Reduction tree phase
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        __syncthreads();
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
		if (s_dataIdx < 2 * blockDim.x)
        {
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
    }
    // 1.2.2. Reverse tree phase
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		__syncthreads();
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
		if (s_dataIdx < 2 * blockDim.x)
        {
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
	}

    // ********** Step 2: Wait to get the sum of the previous block **********
    if (threadIdx.x == 0)
    {
        blkSums[bid] = s_data[2 * blockDim.x - 1];
        if (bid > 0)
        {
            // Wait for the previous block to finish
            while (atomicAdd(&blockCounter2, 0) < bid) {}
            // Read the sum of the previous block
            s_data[blockDim.x * 2] = blkSums[bid - 1];
            // Propagate the partial sum
            blkSums[bid] += s_data[blockDim.x * 2];
            // Memory fence
            __threadfence();
        }
        // Set flag to indicate that the block has finished
        atomicAdd(&blockCounter2, 1);
    }
    __syncthreads();

    // ********** Step 3: Add the previous block's sum to local scan **********
    if (bid > 0)
    {
        s_data[threadIdx.x] += s_data[blockDim.x * 2];
        s_data[threadIdx.x + blockDim.x] += s_data[blockDim.x * 2];
    }

    // ********** Step 4: Write the result to GMEM **********
    if (i1 < n)
    {
        out[i1] = s_data[threadIdx.x];
    }
    if (i2 < n)
    {
        out[i2] = s_data[threadIdx.x + blockDim.x];
    }
}

void exclusiveScanByDevice(int *in, int *out, int n, int blockSize)
{
    printf("\nExclusive scan by device\n");
    GpuTimer timer;
    timer.Start();
    // Allocate device memory
    int *d_in, *d_out, *d_blkSums;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    // Set up grid and block size
    int blkDataSize = 2 * blockSize;
    int numBlocks = (n - 1) / blkDataSize + 1;
    printf("Number of blocks: %d, block size: %d\n", numBlocks, blockSize);
    CHECK(cudaMalloc(&d_blkSums, numBlocks * sizeof(int)));
    
    // Copy data to device
    CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));
    size_t smem = (blkDataSize + 1) * sizeof(int);
    // Call kernel
    BrentKungExclusiveScanKernel<<<numBlocks, blockSize, smem>>>(d_in, d_out, n, d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // Copy result to host
    CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    // Free device memory
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_blkSums));
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

int main()
{
    // int n = 8;
    // int in[] = {1, 9, 5, 1, 6, 4, 7, 2};
    // int correctOut[n];
    // int out[n];
    // // Print the input
    // printf("Input\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     printf("%d ", in[i]);
    // }
    // printf("\n");

    // SET UP INPUT SIZE
    int n = (1 << 26) + 1;
    int blockSize=256;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    int * out = (int *)malloc(bytes); // Device result
    int * correctOut = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = (int)(rand() & 0xFF) - 127; // random int in [-127, 128]
        

    // Scan by host
    exclusiveScanByHost(in, correctOut, n);
    // // Print the correct output
    // printf("Correct output\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     printf("%d ", correctOut[i]);
    // }
    // printf("\n");

    // Scan by device
    exclusiveScanByDevice(in, out, n, blockSize);
    // // Print the output
    // printf("Output\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     printf("%d ", out[i]);
    // }
    // printf("\n");

    // Check correctness
    checkCorrectness(correctOut, out, n);

    return 0;
}