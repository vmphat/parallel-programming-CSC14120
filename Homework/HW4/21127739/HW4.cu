#include <stdio.h>
#include <stdint.h>

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

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

// Kernel function to extract bits
// Each thread extracts 1 bit from 2 elements separated by blockDim.x
__global__ void extractBitsKernel(uint32_t * src, int * bits, int bitIdx, int n)
{
    int i1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (i1 < n)
    {
        bits[i1] = (src[i1] >> bitIdx) & 1;
    }

    int i2 = i1 + blockDim.x;
    if (i2 < n)
    {
        bits[i2] = (src[i2] >> bitIdx) & 1;
    }
}

// Kernel function to compute nOnesBefore
// Do exclusive scan in a single kernel
// Counter to get the dynamic block index for all blocks
__device__ int blockCounter1 = 0;
// Counter to get the sum of all previous blocks
__device__ int blockCounter2 = 0;
// Single-pass exclusive scan by device using Brent-Kung algorithm
__global__ void BrentKungExclusiveScanKernel(int * in, int * out, int n, int * blkSums)
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
    extern __shared__ int s_data[]; // Size of s_data is: 2 * blockDim.x + 1
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
    __syncthreads();

    // 1.2. Each block will scan with data in SMEM
    // 1.2.1. Reduction tree phase
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
		if (s_dataIdx < 2 * blockDim.x)
        {
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
        __syncthreads();
    }
    // 1.2.2. Reverse tree phase
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
		if (s_dataIdx < 2 * blockDim.x)
        {
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        }
        __syncthreads();
	}

    // ********** Step 2: Wait to get the sum of the previous blocks **********
    if (threadIdx.x == 0)
    {
        blkSums[bid] = s_data[2 * blockDim.x - 1];
        if (bid > 0)
        {
            // Wait for the previous block to finish
            while (atomicAdd(&blockCounter2, 0) < bid) {}
            // Read the sum of the previous blocks
            s_data[blockDim.x * 2] = blkSums[bid - 1];
            // Propagate the partial sum
            blkSums[bid] += s_data[blockDim.x * 2];
            // Memory fence to ensure that the sum is written before the flag
            __threadfence();
        }
        // Set flag to indicate that the block has finished
        atomicAdd(&blockCounter2, 1);
    }
    __syncthreads();

    // ********** Step 3: Add the previous blocks' sum to each element **********
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

// Kernel function to compute rank and write to dst
// Each thread processes 2 elements separated by blockDim.x
__global__ void computeRankAndWriteToDstKernel(uint32_t * src, int * bits, int * nOnesBefore, uint32_t * dst, int n)
{
    // Calculate number of zeros in the bits array
    __shared__ int s_nZeros;
    if (threadIdx.x == 0)
    {
        s_nZeros = n - nOnesBefore[n-1] - bits[n-1];
    }
    __syncthreads();
    int nZeros = s_nZeros;

    // Compute rank and write to dst
    int rank;
    int i1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (i1 < n)
    {
        if (bits[i1] == 0)
        {
            rank = i1 - nOnesBefore[i1];
        }
        else
        {
            rank = nZeros + nOnesBefore[i1];
        }
        dst[rank] = src[i1];
    }

    int i2 = i1 + blockDim.x;
    if (i2 < n)
    {
        if (bits[i2] == 0)
        {
            rank = i2 - nOnesBefore[i2];
        }
        else
        {
            rank = nZeros + nOnesBefore[i2];
        }
        dst[rank] = src[i2];
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO

    // Set up gridSize, blockSize, shared memory size for each block
    int blkDataSize = 2 * blockSize; // Each thread processes 2 elements
    int gridSize = (n - 1) / blkDataSize + 1; // Number of blocks in grid
    size_t smemSize = (blkDataSize + 1) * sizeof(int); // +1 for sum of all previous blocks
    int dummyZeroVal = 0; // Dummy value to reset blockCounter1 and blockCounter2

    // [MY_DEBUG] Print out block size, grid size
    printf("Block size: %d, Grid size: %d\n", blockSize, gridSize);

    // Allocate device memories
    // * Source and destination arrays
    uint32_t *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
    // * Bits array and nOnesBefore array
    int *d_bits, *d_nOnesBefore;
    CHECK(cudaMalloc(&d_bits, n * sizeof(int)));
    CHECK(cudaMalloc(&d_nOnesBefore, n * sizeof(int)));
    // * Block sums for exclusive scan in a single kernel
    int *d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize * sizeof(int)));

    // Copy input data from host to device
    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
    // In each loop, sort elements according to the current bit from d_src to d_dst 
    // (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; ++bitIdx)
    {
        // Extract bits parallelly
        extractBitsKernel<<<gridSize, blockSize>>>(d_src, d_bits, bitIdx, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        // Compute nOnesBefore parallelly by using exclusive scan
        // * Reset blockCounter1 and blockCounter2 to 0
        // * Reference: https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_gf268fa2004636b6926fdcd3189152a14.html
        CHECK(cudaMemcpyToSymbol(blockCounter1, &dummyZeroVal, sizeof(int), 0, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(blockCounter2, &dummyZeroVal, sizeof(int), 0, cudaMemcpyHostToDevice));
        // * Call Brent-Kung exclusive scan kernel to compute nOnesBefore
        BrentKungExclusiveScanKernel<<<gridSize, blockSize, smemSize>>>(d_bits, d_nOnesBefore, n, d_blkSums);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        // Compute rank and write to d_dst parallelly
        computeRankAndWriteToDstKernel<<<gridSize, blockSize>>>(d_src, d_bits, d_nOnesBefore, d_dst, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        // Swap d_src and d_dst
        uint32_t *temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }

    // Copy result from device to host
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Free device memories
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_nOnesBefore));
    CHECK(cudaFree(d_blkSums));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    //printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    //printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
