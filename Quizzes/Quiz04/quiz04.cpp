#include <stdio.h>

int main() {
	// Number of elements in the array
	unsigned int N = 1024;
	int blockSize = 128;
	int gridSize = (N + 128 - 1) / 128;

	// Number of threads per block
	printf("Number of threads per block: %d\n", blockSize);
	// Number of blocks in the grid
	printf("Number of blocks in the grid: %d\n", gridSize);

	// Size of float data type
	printf("Size of float data type: %lu\n", sizeof(float));   
	float y_s, b_s[128];
	// Number of byte used for all float variables
	printf("Number of byte used for all float variables: %lu\n", sizeof(y_s) + sizeof(b_s));
}
