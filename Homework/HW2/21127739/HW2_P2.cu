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
#define TILE_WIDTH 32
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

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	//TODO

	/**
	 * Kernel1: Basic Matrix Multiplication
	 *
	 * A naive matrix multiplication kernel assigns one thread 
	 * to compute one element of the output matrix C.
	 * 
	 * Each thread loads one row of matrix A and one column of 
	 * matrix B from global memory, do the dot product, and 
	 * store the result back to matrix C in the global memory.
	 *
	 * Reference: https://github.com/lzhengchun/matrix-cuda
	**/
	
	// Calculate the row and column index of the element
	// that this thread is responsible for
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the thread is within the matrix bounds
	if (row < m && col < k)
	{
		// Compute the dot product of the row (of matrix A)
		// and column (of matrix B)
		float sum = 0.0f;

		// Iterate over the shared dimension
		for (int i = 0; i < n; ++i)
		{
			sum += A[row * n + i] * B[i * k + col];
		}

		// Store the result in matrix C at position (row, col)
		C[row * k + col] = sum;
	}
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO

	/**
	 * Kernel2: Tiled Matrix Multiplication
	 *
	 * A more optimized matrix multiplication kernel that uses 
	 * shared memory to reduce the number of global memory accesses.
	 * 
	 * One thread block computes one tile of the output matrix C.
	 * Each thread in the block computes one element of the tile.
	 * This kernel computes C in multiple iterations. In each iteration,
	 * each thread block loads one tile of A and one tile of B from 
	 * global memory into shared memory, performs computation, and 
	 * stores temporary results of C in registers. After all iterations
	 * are done, the thread block stores one tile of C back to global memory.
	 * 
	 * Reference: https://github.com/lzhengchun/matrix-cuda
	**/

	// Calculate the row and column index of the element
	// that this thread is responsible for
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	// Accumulate the result for C[row, col] in a register
	float sum = 0.0f;

	// Split the computation into multiple tiles, each tile
	// handles TILE_WIDTH columns of matrix A and TILE_WIDTH
	// rows of matrix B
	for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
	{
		// Load one tile of matrix A and B from global memory
		// into shared memory
		if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
		{
			s_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
		}
		else
		{
			s_A[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if ((t * TILE_WIDTH + threadIdx.y) < n && col < k)
		{
			s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * k + col];
		}
		else
		{
			s_B[threadIdx.y][threadIdx.x] = 0.0f;
		}

		// Synchronize to make sure that all threads within the
		// block have loaded the tiles into shared memory
		__syncthreads();

		// Compute the dot product for the current tile from s_A and s_B
		for (int i = 0; i < TILE_WIDTH; ++i)
		{
			sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
		}

		// Synchronize to make sure that all threads have finished
		// computing the current tile before loading the next tile
		// in the next iteration
		__syncthreads();
	}

	// Check if the thread is within the matrix bounds and 
	// store the result back to matrix C in the global memory
	if (row < m && col < k)
	{
		C[row * k + col] = sum;
	}
}

void matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
    bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1)
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        // TODO
		
		/**
		 * Matrix multiplication on the host (CPU)
		**/

		// Iterate over rows of matrix A
		for (int row = 0; row < m; ++row)
		{
			// Iterate over columns of matrix B
			for (int col = 0; col < k; ++col)
			{
				// Compute the dot product of the row (of matrix A)
				// and column (of matrix B)
				float sum = 0.0f;

				// Iterate over the shared dimension
				for (int i = 0; i < n; ++i)
				{
					sum += A[row * n + i] * B[i * k + col];
				}

				// Store the result in matrix C at position (row, col)
				C[row * k + col] = sum;
			}
		}
    }
    else // Use device
    {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
		// Calculate the size of each matrix in bytes
		size_t sizeA = m * n * sizeof(float);
		size_t sizeB = n * k * sizeof(float);
		size_t sizeC = m * k * sizeof(float);
		// Allocate device memories
		CHECK(cudaMalloc((void **)&d_A, sizeA));
		CHECK(cudaMalloc((void **)&d_B, sizeB));
		CHECK(cudaMalloc((void **)&d_C, sizeC));

        // TODO: Copy data to device memories
		CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));
        
        dim3 gridSize(1); // TODO: Compute gridSize
		gridSize.x = (k - 1) / blockSize.x + 1;
		gridSize.y = (m - 1) / blockSize.y + 1;
        
		if (kernelType == 1)
			matrix_multiplication_kernel1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
		else if (kernelType == 2)
			matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        // TODO: Copy result from device memory
		// Synchonize and check for errors during the kernel launch
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
		// Copy the result from the device memory to the host
		CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

        // TODO: Free device memories
		CHECK(cudaFree(d_A));
		CHECK(cudaFree(d_B));
		CHECK(cudaFree(d_C));
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x,gridSize.y, blockSize.x,blockSize.y);

    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
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
int main(int argc, char** argv)
{
	printDeviceInfo();
	
	//Declare variables
    float* h_A; // The A matrix
    float* h_B; // The B matrix
    float* h_C; // The output C matrix
    float* correct_C; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B
    int k; // number of columns in the matrix B

    m = (1 << 10);
    n = (1 << 9);
    k = (1 << 10);

    // Set up input data
    h_A = (float*)malloc(m * n * sizeof(float));
    h_B = (float*)malloc(n * k * sizeof(float));
    h_C = (float*)malloc(m * k * sizeof(float));
    correct_C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0;j < n;j++)
            h_A[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
 
    for (int i = 0; i < n; i++)
        for (int j = 0;j < k;j++)
            h_B[i*k+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);


    // Add vectors (on host)
    matrix_multiplication(h_A,h_B,correct_C,m,n,k);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    // Add in1 & in2 on device
	printf("Basic Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,1);
	float err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f\n\n", err);

	printf("Shared memory Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,2);
	err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f", err);	
	
    free(h_A);
    free(h_B);
    free(h_C);
    free(correct_C);

    return 0;
}
