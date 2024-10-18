/*‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾*\
|   Họ và tên: Vũ Minh Phát                     |
|   MSSV: 21127739                              |
|   Lớp: Lập trình song song - 21KHMT           |
|   HW0: Làm quen với CUDA                      |
+-----------------------------------------------+
|   Câu 2: Viết chương trình cộng hai vector,   |
|           mỗi thread thực hiện hai phép tính  |
|           cộng trên hai phần tử của mảng.     |
\*_____________________________________________*/

#include <stdio.h>
// #include <cuda_runtime.h>

// // Macro để kiểm tra lỗi sau khi gọi hàm CUDA API.
// // Nếu có lỗi, nó sẽ in thông báo lỗi và thoát chương trình.
// #define CUDA_CHECK_ERROR(call) {                                \
//     const cudaError_t err = call;                               \
//     if (err != cudaSuccess) {                                   \
//         fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);  \
//         fprintf(stderr, "code: %d, reason: %s\n",               \
//                 err, cudaGetErrorString(err));                  \
//         exit(EXIT_FAILURE);                                     \
//     }                                                           \
// }

// // Kích thước block được thống nhất để chạy thử nghiệm
// #define BLOCK_SIZE 256

// /**
//  * Cấu trúc để tính thời gian chạy của chương trình
//  *
//  * Reference: File demo `01-AddVector.cu` được 
//  *              cung cấp trong môn học.
//  */
// struct GpuTimer {
// 	cudaEvent_t start;
// 	cudaEvent_t stop;

//     // Khởi tạo cấu trúc GpuTimer
// 	GpuTimer() {
// 		cudaEventCreate(&start);
// 		cudaEventCreate(&stop);
// 	}

//     // Hủy cấu trúc GpuTimer
// 	~GpuTimer() {
// 		cudaEventDestroy(start);
// 		cudaEventDestroy(stop);
// 	}

//     // Bắt đầu tính thời gian
// 	void Start() {
// 		cudaEventRecord(start, 0);                                                                 
// 		cudaEventSynchronize(start);
// 	}

//     // Kết thúc tính thời gian
// 	void Stop() {
// 		cudaEventRecord(stop, 0);
// 	}

//     // Trả về thời gian chênh lệch giữa `start` và `stop`
// 	float Elapsed() {
// 		float elapsed;
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&elapsed, start, stop);
// 		return elapsed;
// 	}
// };

// /**
//  * Hàm cộng hai vector trên host (CPU)
//  *
//  * @param in1: con trỏ đến vector thứ nhất
//  * @param in2: con trỏ đến vector thứ hai
//  * @param out: con trỏ đến vector lưu kết quả
//  * @param n: số phần tử của mỗi vector
//  */
// void addVecOnHost(const float* in1, const float* in2, float* out, const int& n) {
//     for (int i = 0; i < n; ++i) {
//         out[i] = in1[i] + in2[i];
//     }
// }

#define BLOCK_SIZE 3
// Định nghĩa số lượng phần tử mà mỗi thread sẽ xử lý
#define ELEMENTS_PER_THREAD 3


void AddVectorV1(float* in1, float* in2, float* out, int n,
                 int blockIdx, int threadIdx
  ) {
    // Mỗi block xử lý 2 * BLOCK_SIZE phần tử
    size_t elementsPerBlock = ELEMENTS_PER_THREAD * BLOCK_SIZE;
    size_t blockStartIdx    = blockIdx * elementsPerBlock;
    size_t threadStartIdx   = blockStartIdx + threadIdx;

    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        size_t idx = threadStartIdx + i * BLOCK_SIZE;
        if (idx >= n) { return; }
        out[idx] = in1[idx] + in2[idx];
        printf("Thread Index %d, Data Index: %lu\n", threadIdx, idx);
    }

    // // Giai đoạn 1: Xử lý BLOCK_SIZE phần tử đầu
    // size_t idx1 = blockStartIdx + threadIdx;
    // if (idx1 >= n) { return; }
    // out[idx1] = in1[idx1] + in2[idx1];
    // printf("Thread Index %d, Data Index: %lu\n", threadIdx, idx1);

    // // Giai đoạn 2: Xử lý BLOCK_SIZE phần tử sau
    // size_t idx2 = idx1 + BLOCK_SIZE;
    // if (idx2 >= n) { return; }
    // out[idx2] = in1[idx2] + in2[idx2];
    // printf("Thread Index %d, Data Index: %lu\n", threadIdx, idx2);
}


void AddVectorV2(float* in1, float* in2, float* out, int n,
                 int blockIdx, int threadIdx
  ) {
    // Index toàn cục của thread này
    const size_t globalThreadIndex = blockIdx * BLOCK_SIZE + threadIdx;
    // Index của phần tử đầu tiên mà thread này xử lý
    const size_t threadStartIndex  = globalThreadIndex * ELEMENTS_PER_THREAD;

    for (size_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        const size_t index = threadStartIndex + i;
        if (index >= n) { return; }
        out[index] = in1[index] + in2[index];
        printf("Thread Index %d, Data Index: %lu\n", threadIdx, index);
    }

}

// Enum để chọn loại thiết bị (host hoặc device)
enum DeviceType {
    HOST,           // Chạy trên host (CPU)
    DEVICE,         // Chạy trên device (GPU)
    NUM_DEVICES     // Số loại thiết bị hiện có
};
// Enum để chọn version của kernel (chỉ dùng cho device)
enum KernelVersion {
    VERSION_1,      // Gọi hàm kernel `addVecOnDeviceV1`
    VERSION_2,      // Gọi hàm kernel `addVecOnDeviceV2`
    NUM_VERSIONS    // Số version hiện có
};

#define INNER_LINE printf("+-----------------+-----------------+-----------------+-----------------+\n")
#define OUTER_LINE printf("*-------------------------------------------------------------------------*\n")

int main() {
    float in1[10] = {0};
    float in2[10] = {0};
    float out[10] = {0};
    int n = sizeof(in1) / sizeof(in1[0]);

    for (int i = 0; i < n; ++i) {
        in1[i] = i;
        in2[i] = n-i;
    }

    printf("Input 1: "); for (int i = 0; i < n; ++i) { printf("%-5.1f ", in1[i]); } printf("\n");
    printf("Input 2: "); for (int i = 0; i < n; ++i) { printf("%-5.1f ", in2[i]); } printf("\n");

    // Tính số lượng block cần thiết
    // Tính số block cần thiết
    size_t elementsPerBlock = ELEMENTS_PER_THREAD * BLOCK_SIZE;
    size_t numBlocks = (n + elementsPerBlock - 1) / elementsPerBlock;
    printf("Number of blocks: %lu\n", numBlocks);

    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        printf("Block %d\n", blockIdx);
        for (int threadIdx = 0; threadIdx < BLOCK_SIZE; ++threadIdx) {
            AddVectorV2(in1, in2, out, n, blockIdx, threadIdx);
        }
    }
    printf("Output: "); for (int i = 0; i < n; ++i) { printf("%-5.1f ", out[i]); } printf("\n");


    INNER_LINE;
    printf("*-------------------------------------------------------------------------*\n");
    printf("| %-12s| %-18s| %-18s| %-18s|\n", "Vector size", " Host time (ms)", "Device time (ms)", "Device time (ms)");
    printf("| %-12s| %-18s| %-18s| %-18s|\n", " ", " ", "   (Version 1)", "   (Version 2)");
    printf("+-------------------------------------------------------------------------+\n");
}
