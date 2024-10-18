#include <stdio.h>

int main() {
    unsigned int M = 150;
    unsigned int N = 300;

    int bd_x = 16;
    int bd_y = 32;
    int bd_z = 1;

    int gd_x = (N - 1) / 16 + 1;
    int gd_y = (M - 1) / 32 + 1;
    int gd_z = 1;

    printf("M = %d, N = %d\n", M, N);
    printf("bd: x = %d, y = %d, z = %d\n", bd_x, bd_y, bd_z);
    printf("gd: x = %d, y = %d, z = %d\n", gd_x, gd_y, gd_z);
    printf("blockDim bd: x * y * z = %d\n", bd_x * bd_y * bd_z);
    printf("gridDim  gd: x * y * z = %d\n", gd_x * gd_y * gd_z);
    printf("total threads = %d\n", bd_x * bd_y * bd_z * gd_x * gd_y * gd_z);
    
    printf("*--------------------------------------*\n");
    // Number of threads per block?
    int num_thread_per_block = bd_x * bd_y * bd_z;
    printf("Number of threads per block = %d\n", num_thread_per_block);

    //  number of blocks in the grid?
    int num_block = gd_x * gd_y * gd_z;
    printf("Number of blocks in the grid = %d\n", num_block);

    // number of threads in the grid?
    int num_thread_in_grid = num_thread_per_block * num_block;
    printf("Number of threads in the grid = %d\n", num_thread_in_grid);

    // Number of thread with  (row < M && col < N)
    int num_thread = M * N;
    printf("Number of thread with  (row < M && col < N) = %d\n", num_thread);


    printf("*--------------------------------------*\n");
    int width = 400, height = 500;
    int row = 20, col = 10;
    // array index of the matrix in row-major order
    int index = row * width + col;
    printf("array index of the matrix in row-major order = %d\n", index);

    // array index of the matrix in column-major order
    index = col * height + row;
    printf("array index of the matrix in column-major order = %d\n", index);


    printf("*--------------------------------------*\n");
    int new_width = 400, new_height = 500, new_depth = 300;
    int new_row = 10, new_col = 20, z = 5;
    // tensor is stored as a one-dimensional array in row-major order
    // Specify the array index
    // int new_index = z * new_width * new_height + new_row * new_width + new_col;
    // printf("array index of the tensor in row-major order = %d\n", new_index);

    int new_index = (z * new_height * new_width) + (new_row * new_width) + new_col;
    printf("array index of the tensor in row-major order = %d\n", new_index);
}