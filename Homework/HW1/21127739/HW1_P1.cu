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

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  

    // Calculate the row and column index of the pixel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	// If the pixel coordinate is within the image
    if (row < height && col < width)
    {
		// Get 1D offset for the grayscale image
		int grayOffset = row * width + col;

        // Get 1D offset for the RGB image
		// Each pixel in RGB image has 3 values: red, green, and blue
		int rgbOffset = 3 * grayOffset;
		// Get red, green, and blue values of the pixel
        uint8_t red   = inPixels[rgbOffset    ];
        uint8_t green = inPixels[rgbOffset + 1];
        uint8_t blue  = inPixels[rgbOffset + 2];

		// Perform the rescaling and store the result
        outPixels[grayOffset] = 0.299f*red + 0.587f*green + 0.114f*blue;
    }
}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}
	else // use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO: Allocate device memories
		uint8_t *d_inPixels, *d_outPixels;
		// Number of bytes for the RGB image and the grayscale image
		size_t numBytesRgb  = width * height * 3 * sizeof(uint8_t);
		size_t numBytesGray = width * height * sizeof(uint8_t);
        CHECK(cudaMalloc((void **)&d_inPixels , numBytesRgb ));
        CHECK(cudaMalloc((void **)&d_outPixels, numBytesGray));

		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_inPixels, inPixels, numBytesRgb, cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel (remember to check kernel error)
        dim3 gridSize((width  - 1) / blockSize.x + 1, 
                      (height - 1) / blockSize.y + 1);
        convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outPixels);
		// Ensure all device operations are complete
		// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
		CHECK(cudaDeviceSynchronize());
		// Check for kernel launch errors
		// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR
		CHECK(cudaGetLastError());

		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(outPixels, d_outPixels, numBytesGray, cudaMemcpyDeviceToHost));

		// TODO: Free device memories
        CHECK(cudaFree(d_inPixels));
        CHECK(cudaFree(d_outPixels));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += abs((int)a1[i] - (int)a2[i]);
	err /= n;
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{	
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * correctOutPixels= (uint8_t *)malloc(width * height);
	convertRgb2Gray(inPixels, width, height, correctOutPixels);

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	} 
	convertRgb2Gray(inPixels, width, height, outPixels, true, blockSize); 

	// Compute mean absolute error between host result and device result
	float err = computeError(outPixels, correctOutPixels, width * height);
	printf("Error between device result and host result: %f\n", err);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(outPixels);
}
