/*‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾*\
|   Họ và tên: Vũ Minh Phát                     |
|   MSSV: 21127739                              |
|   Lớp: Lập trình song song - 21KHMT           |
|   HW0: Làm quen với CUDA                      |
+-----------------------------------------------+
|   Câu 1: Viết hàm và thử nghiệm in ra các     |
|           thông tin của card màn hình.        |
\*_____________________________________________*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro để kiểm tra lỗi sau khi gọi hàm CUDA API.
// Nếu có lỗi, nó sẽ in thông báo lỗi và thoát chương trình.
#define CUDA_CHECK_ERROR(call) {                                \
    const cudaError_t err = call;                               \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);  \
        fprintf(stderr, "code: %d, reason: %s\n",               \
                err, cudaGetErrorString(err));                  \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

/**
 * Hàm in ra các thông tin của card màn hình (GPU), bao gồm:
 *  - GPU card’s name
 *  - GPU computation capabilities
 *  - Maximum number of block dimensions
 *  - Maximum number of grid dimensions
 *  - Maximum size of GPU memory
 *  - Amount of constant and share memory
 *  - Warp size
 *
 * @param device: Số thứ tự của GPU cần in thông tin
 *
 * References: 
 * [1] Thông tin của cấu trúc `cudaDeviceProp`: 
 *      https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
 * [2] Gist "Get Information about CUDA cards on your system"
 *      by Steven Borrelli: https://gist.github.com/stevendborrelli/4286842
 */
void printGPUInfo(int device) {
    cudaDeviceProp deviceProp;

    // Dùng `cudaGetDeviceProperties()` để lấy thông tin của GPU
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, device));

    printf("===== GPU Information =====\n");

    // GPU card's name
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c:~:text=ASCII%20string%20identifying%20device
    printf("GPU name: %s\n", deviceProp.name);

    // GPU computation capabilities
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0:~:text=device%27s%20compute%20capability
    printf("Compute capability: %d.%d\n", 
           deviceProp.major, deviceProp.minor);

    // Maximum number of block dimensions
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_192d195493a9d36b2d827aaf3ffd89f1e:~:text=Maximum%20size%20of%20each%20dimension%20of%20a%20block
    printf("Maximum block dimensions: (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);

    // Maximum number of grid dimensions
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17d138a572315b3bbb6caf7ccc914a130:~:text=Maximum%20size%20of%20each%20dimension%20of%20a%20grid
    printf("Maximum grid dimensions: (%d, %d, %d)\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

    // Global memory available on device (in GB)
    // Refence: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1983c292e2078dd5a4240f49c41d647f3:~:text=Global%20memory%20available%20on%20device%20in%20bytes
    printf("Total global memory: %.2f GB\n",
           (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));

    // Constant memory available on device (in KB)
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1983c292e2078dd5a4240f49c41d647f3:~:text=Constant%20memory%20available%20on%20device%20in%20bytes
    printf("Total constant memory: %.2f KB\n",
           (float)deviceProp.totalConstMem / 1024);
    
    // Shared memory available per block (in KB)
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_182ec4c5e244addb9cd57f5b9da0eaca7:~:text=Shared%20memory%20available%20per%20block%20in%20bytes
    printf("Total shared memory per block: %.2f KB\n",
           (float)deviceProp.sharedMemPerBlock / 1024);
    // Shared memory available per multiprocessor (in KB)
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_182ec4c5e244addb9cd57f5b9da0eaca7:~:text=Shared%20memory%20available%20per%20multiprocessor%20in%20bytes
    printf("Total shared memory per multiprocessor: %.2f KB\n",
           (float)deviceProp.sharedMemPerMultiprocessor / 1024);

    // Warp size in threads
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_18656f53eb2a7e54500f6fb95a830b47d:~:text=Warp%20size%20in%20threads
    printf("Warp size: %d\n", deviceProp.warpSize);

    printf("===========================\n\n");
}

int main(int argc, char **argv) {
    int deviceCount = 0;

    // Lấy số lượng GPU có trong hệ thống
    // Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));

    // Nếu không tìm thấy GPU nào, in thông báo và thoát
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    // In số lượng GPU có trong hệ thống
    printf("Number of CUDA-capable devices: %d\n\n", deviceCount);

    // Duyệt qua từng GPU và in ra thông tin tương ứng
    for (int device = 0; device < deviceCount; ++device) {
        printf("Device %d:\n", device);
        printGPUInfo(device);
    }

    return EXIT_SUCCESS;
}
