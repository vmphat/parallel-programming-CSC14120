#include <stdio.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
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


__global__ void reduceBlksKernel1(int * in, int * out, int n)
{
	// TODO

	// Kernel function – 1st version: A simple approach (Interleaved addressing)
	// Reference: Slide "Lecture04-CUDAParallelExecution(P2)", page 15

	// Number of elements processed by all previous blocks
	int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;
	// Index of the first element processed by this thread
	int threadOffset = numElemsBeforeBlk + threadIdx.x * 2;

	for (int stride = 1; stride < blockDim.x * 2; stride *= 2)
	{
		// Only activate threads whose index is a multiple of stride
		if (threadIdx.x % stride == 0)
		{
			// Check if there is a valid "next" element to sum
			if (threadOffset + stride < n)
			{
				in[threadOffset] += in[threadOffset + stride];
			}
		}
		// Synchronize threads within each block
		__syncthreads();
	}

	// Add the local sum of this block to the global sum
	// Only the first thread of each block will do this
	if (threadIdx.x == 0)
	{
		// Use atomic operation to prevent race conditions 
		// when updating the global sum
		// Reference: http://supercomputingblog.com/cuda/cuda-tutorial-4-atomic-operations/
		atomicAdd(out, in[numElemsBeforeBlk]);
	}
}

__global__ void reduceBlksKernel2(int * in, int * out,int n)
{
	// TODO

	// Kernel function – 2nd version: Reduce warp divergence
	// Reference: Slide "Lecture04-CUDAParallelExecution(P2)", page 20

	// Number of elements processed by all previous blocks
	int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;

	for (int stride = 1; stride < blockDim.x * 2; stride *= 2)
	{
		// Only activate the first few threads of each block
		if (threadIdx.x < blockDim.x / stride)
		{
			// Index of the first element processed by this thread
			int threadOffset = numElemsBeforeBlk + threadIdx.x * 2 * stride;
			
			// Check if there is a valid "next" element to sum
			if (threadOffset + stride < n)
			{
				in[threadOffset] += in[threadOffset + stride];
			}
		}
		// Synchronize threads within each block
		__syncthreads();
	}
	
	// Add the local sum of this block to the global sum
	// Only the first thread of each block will do this
	if (threadIdx.x == 0)
	{
		// Use atomic operation to prevent race conditions 
		// when updating the global sum
		// Reference: http://supercomputingblog.com/cuda/cuda-tutorial-4-atomic-operations/
		atomicAdd(out, in[numElemsBeforeBlk]);
	}
}

__global__ void reduceBlksKernel3(int * in, int * out,int n)
{
	// TODO

	// Kernel function – 3rd version: Reduce warp divergence + Sequential addressing
	// Reference: Slide "Lecture04-CUDAParallelExecution(P2)", page 21

	// Number of elements processed by all previous blocks
	int numElemsBeforeBlk = blockIdx.x * blockDim.x * 2;

	for (int stride = blockDim.x; stride > 0; stride /= 2)
	{
		// Only activate the first few threads of each block
		if (threadIdx.x < stride)
		{
			// Index of the first element processed by this thread
			int threadOffset = numElemsBeforeBlk + threadIdx.x;
			
			// Check if there is a valid "next" element to sum
			if (threadOffset + stride < n)
			{
				in[threadOffset] += in[threadOffset + stride];
			}
		}
		// Synchronize threads within each block
		__syncthreads();
	}
	
	// Add the local sum of this block to the global sum
	// Only the first thread of each block will do this
	if (threadIdx.x == 0)
	{
		// Use atomic operation to prevent race conditions 
		// when updating the global sum
		// Reference: http://supercomputingblog.com/cuda/cuda-tutorial-4-atomic-operations/
		atomicAdd(out, in[numElemsBeforeBlk]);
	}
}

int reduce(int const * in, int n,
        bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1)
{

	GpuTimer timer;
	int result = 0; // Init
	if (useDevice == false)
	{
		timer.Start();
		result = in[0];
		for (int i = 1; i < n; i++)
		{
			result += in[i];
		}
		timer.Stop();
		float hostTime = timer.Elapsed();
		printf("Host time: %f ms\n",hostTime);
	}
	else // Use device
	{
		// Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize(1); // TODO: Compute gridSize from n and blockSize
		
		// Number of elements each block will process
		int numElemsPerBlk = blockSize.x * 2;
		// Number of blocks needed to process all elements
		// (n - 1) / (blockSize.x * 2) + 1
		gridSize.x = (n - 1) / numElemsPerBlk + 1;

		// TODO: Allocate device memories
		CHECK(cudaMalloc((void **)&d_in , n * sizeof(int)));
		CHECK(cudaMalloc((void **)&d_out, sizeof(int)));

		// TODO: Copy data to device memories
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// Call kernel
		timer.Start();
		if (kernelType == 1)
			reduceBlksKernel1<<<gridSize, blockSize>>>(d_in, d_out, n);
		else if (kernelType == 2)
			reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, d_out, n);
		else 
			reduceBlksKernel3<<<gridSize, blockSize>>>(d_in, d_out, n);

		cudaDeviceSynchronize();
		timer.Stop();
		float kernelTime = timer.Elapsed();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
		CHECK(cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost));

		// TODO: Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));

		// Print info
		printf("\nKernel %d\n", kernelType);
		printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
		printf("Kernel time = %f ms\n", kernelTime);
	}

	return result;
}

void checkCorrectness(int r1, int r2)
{
	if (r1 == r2)
		printf("CORRECT :)\n");
	else
		printf("INCORRECT :(\n");
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
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Set up input size
    int n = (1 << 24)+1;
    printf("Input size: %d\n", n);

    // Set up input data
    int * in = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce NOT using device
    int correctResult = reduce(in, n);

    // Reduce using device, kernel1
    dim3 blockSize(512); // Default
    if (argc == 2)
    	blockSize.x = atoi(argv[1]); 
 	
	int result1 = reduce(in, n, true, blockSize, 1);
    checkCorrectness(result1, correctResult);

    // Reduce using device, kernel2
    int result2 = reduce(in, n, true, blockSize, 2);
    checkCorrectness(result2, correctResult);

    // Reduce using device, kernel3
    int result3 = reduce(in, n, true, blockSize, 3);
    checkCorrectness(result3, correctResult);

    // Free memories
    free(in);
}
