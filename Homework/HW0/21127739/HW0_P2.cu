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

// Kích thước block được thống nhất để chạy thử nghiệm
#define BLOCK_SIZE 256
// Số phần tử mỗi thread xử lý
#define ELEMENTS_PER_THREAD 2
// Giá trị dummy để "làm nóng" GPU
#define DUMMY_VALUE 1
// In các dòng kẻ của bảng kết quả
#define OUTER_LINE printf("*-------------*-------------------*-------------------*-------------------*\n");
#define INNER_LINE printf("*-------------+-------------------+-------------------+-------------------*\n");

// Enum để chọn loại thiết bị (host hoặc device)
enum DeviceType {
    HOST,       // Chạy trên host (CPU)
    DEVICE,     // Chạy trên device (GPU)
};
// Enum để chọn version của kernel (chỉ dùng cho device)
enum KernelVersion {
    VERSION_1,  // Gọi hàm kernel `addVecOnDeviceV1`
    VERSION_2,  // Gọi hàm kernel `addVecOnDeviceV2`
    NONE,       // Không chọn version nào
};

/**
 * Cấu trúc để tính thời gian chạy của chương trình
 *
 * Reference: File demo `01-AddVector.cu` được 
 *              cung cấp trong môn học.
 */
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    // Khởi tạo cấu trúc GpuTimer
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // Hủy cấu trúc GpuTimer
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Bắt đầu tính thời gian
    void Start() {
        cudaEventRecord(start, 0);                                                                 
        cudaEventSynchronize(start);
    }

    // Kết thúc tính thời gian
    void Stop() {
        cudaEventRecord(stop, 0);
    }

    // Trả về thời gian chênh lệch giữa `start` và `stop`
    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

/**
 * Hàm cộng hai mảng (vector) trên host (CPU).
 *
 * Ta dùng một vòng lặp để cộng từng phần tử của 
 *  hai mảng và lưu kết quả vào mảng thứ ba.
 *
 * @param in1: con trỏ đến mảng thứ nhất
 * @param in2: con trỏ đến mảng thứ hai
 * @param out: con trỏ đến mảng lưu kết quả
 * @param n: số phần tử của mỗi mảng
 */
void addVecOnHost(const float *in1, const float *in2, float *out, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = in1[i] + in2[i];
    }
}

/**
 * Hàm cộng hai mảng (vector) trên device (GPU) theo version 1.
 * 
 * Mỗi thread block xử lý `ELEMENTS_PER_THREAD * blockDim.x` 
 *  phần tử liên tiếp. Tất cả các thread trong mỗi block sẽ xử lý
 *  `blockDim.x` phần tử đầu mảng, mỗi thread xử lý một phần tử. 
 *  Sau đó tất cả các thread sẽ lần lượt chuyển sang `blockDim.x`
 *  phần tử tiếp theo của mảng, mỗi thread xử lý một phần tử.
 *  Quá trình này lặp lại cho đến khi hết mảng. 
 *
 * @param in1: con trỏ đến mảng thứ nhất trên device
 * @param in2: con trỏ đến mảng thứ hai trên device
 * @param out: con trỏ đến mảng lưu kết quả trên device
 * @param n: số phần tử của mỗi mảng
 */
__global__ 
void addVecOnDeviceV1(const float *in1, const float *in2, float *out, const size_t n) {
    // Số lượng phần tử mỗi (thread-)block có thể xử lý
    size_t elementsPerBlock = ELEMENTS_PER_THREAD * blockDim.x;
    // Index của phần tử đầu tiên mà (thread-)block này xử lý
    size_t blockStartIndex  = blockIdx.x * elementsPerBlock;
    // Index của phần tử đầu tiên mà thread này xử lý
    size_t threadStartIndex = blockStartIndex + threadIdx.x;

    // Mỗi thread xử lý `ELEMENTS_PER_THREAD` phần tử được mô tả như sau:
    //  - Phần tử thứ 1: `threadStartIndex`
    //  - Phần tử thứ 2: `threadStartIndex + blockDim.x`
    //  - Phần tử thứ 3: `threadStartIndex + 2 * blockDim.x`
    //  - ...
    for (size_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        size_t index = threadStartIndex + i * blockDim.x;
        if (index >= n) { return; }
        out[index] = in1[index] + in2[index];
    }
}

/**
 * Hàm cộng hai mảng (vector) trên device (GPU) theo version 2.
 *
 * Mỗi thread block xử lý `ELEMENTS_PER_THREAD * blockDim.x` phần tử 
 *  liên tiếp. Mỗi thread sẽ xử lý `ELEMENTS_PER_THREAD` phần tử 
 *  liên tiếp nhau trong mảng.
 *
 * @param in1: con trỏ đến mảng thứ nhất trên device
 * @param in2: con trỏ đến mảng thứ hai trên device
 * @param out: con trỏ đến mảng lưu kết quả trên device
 * @param n: số phần tử của mỗi mảng
 */
__global__ 
void addVecOnDeviceV2(const float *in1, const float *in2, float *out, const size_t n) {
    // Global index của thread này (tương đương với index của phần tử mảng)
    size_t globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // Index của phần tử đầu tiên mà thread này xử lý
    size_t threadStartIndex  = globalThreadIndex * ELEMENTS_PER_THREAD;

    // Mỗi thread xử lý `ELEMENTS_PER_THREAD` phần tử được mô tả như sau:
    //  - Phần tử thứ 1: `threadStartIndex`
    //  - Phần tử thứ 2: `threadStartIndex + 1`
    //  - Phần tử thứ 3: `threadStartIndex + 2`
    //  - ...
    for (size_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        size_t index = threadStartIndex + i;
        if (index >= n) { return; }
        out[index] = in1[index] + in2[index];
    }
}

float addVec(const float *in1, const float *in2, float *out, const size_t n,
             DeviceType device, KernelVersion version = KernelVersion::NONE) {
    // Khởi tạo timer để đo thời gian chạy
    GpuTimer timer;

    switch(device) {
    // ==================== CHẠY TRÊN HOST ====================
    case DeviceType::HOST: {
        timer.Start();
        addVecOnHost(in1, in2, out, n);
        timer.Stop();
        break;
    }
    // =================== CHẠY TRÊN DEVICE ===================
    case DeviceType::DEVICE: {
        // Host cấp phát bộ nhớ cho các mảng trên device
        float *d_in1, *d_in2, *d_out;
        size_t nBytes = n * sizeof(float);
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_in1, nBytes));
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_in2, nBytes));
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_out, nBytes));
        
        // Copy dữ liệu của mảng đầu vào từ host sang device
        CUDA_CHECK_ERROR(cudaMemcpy(d_in1, in1, nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_in2, in2, nBytes, cudaMemcpyHostToDevice));

        // Xác định hàm kernel cần chạy
        void (*kernelFunc)(const float *, const float *, float *, const size_t);
        if (version == KernelVersion::VERSION_1) {
            kernelFunc = addVecOnDeviceV1;
        } 
        else if (version == KernelVersion::VERSION_2) {
            kernelFunc = addVecOnDeviceV2;
        }
        else {
            fprintf(stderr, "Error: Invalid kernel version\n");
            exit(EXIT_FAILURE);
        }

        // Xác định kích thước của grid và block
        size_t elementsPerBlock = ELEMENTS_PER_THREAD * BLOCK_SIZE;
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((n - 1) / elementsPerBlock + 1);

        // Chạy hàm kernel và đo thời gian chạy
        timer.Start();
        kernelFunc<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);
        cudaDeviceSynchronize(); // Host waits here until device completes its work
        timer.Stop();
        
        // Copy kết quả của mảng đầu ra từ device trở lại host
        CUDA_CHECK_ERROR(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

        // Host giải phóng bộ nhớ cho các mảng trên device sau khi kết thúc
        CUDA_CHECK_ERROR(cudaFree(d_in1));
        CUDA_CHECK_ERROR(cudaFree(d_in2));
        CUDA_CHECK_ERROR(cudaFree(d_out));
        
        break;
    }
    // ============== LOẠI THIẾT BỊ KHÔNG HỢP LỆ ==============
    default: {
        fprintf(stderr, "Error: Invalid device type\n");
        exit(EXIT_FAILURE);
    }
    }

    // Trả về thời gian chạy
    return timer.Elapsed();
}

int main(int argc, char **argv) {
    // Các kích thước mảng (N) khác nhau được dùng để thử nghiệm
    const size_t VECTOR_SIZES[] = {
        DUMMY_VALUE, // Dùng giá trị dummy để "làm nóng" GPU
        64, 256, 1024, 4096, 16384, 
        65536, 262144, 1048576, 4194304, 16777216
    };

    // Số trường hợp cần thử nghiệm
    size_t nTestCases = sizeof(VECTOR_SIZES) / sizeof(VECTOR_SIZES[0]);
    // Khởi tạo các mảng chứa kết quả của từng trường hợp thử nghiệm
    float *hostTimes     = (float *)malloc(nTestCases * sizeof(float));
    float *deviceTimesV1 = (float *)malloc(nTestCases * sizeof(float));
    float *deviceTimesV2 = (float *)malloc(nTestCases * sizeof(float));
    

    // +========================================================+
    // |             In thông tin của card màn hình             |
    // +--------------------------------------------------------+
    cudaDeviceProp deviceProp;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    printf("===== GPU Information =====\n");
    printf("GPU name: %s\n", deviceProp.name);
    printf("GPU compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("===========================\n\n");


    // +========================================================+
    // |   Chạy thử nghiệm với các kích thước mảng khác nhau    |
    // +--------------------------------------------------------+
    printf("________[ Running test cases ]________\n");
    for (size_t caseIdx = 0; caseIdx < nTestCases; ++caseIdx) {
        // Kích thước mảng đầu vào
        size_t N = VECTOR_SIZES[caseIdx];
        fprintf(stderr, "Vector size N = %zu ... ", N);

        /**
         * Cấp phát bộ nhớ cho các mảng trên host
         *  - `in1`, `in2`: mảng đầu vào
         *  - `correctOut`: mảng lưu kết quả đúng (tính toán trên host)
         *  - `outV1`, `outV2`: mảng lưu kết quả tính toán trên device
         */
        size_t nBytes     = N * sizeof(float);
        float *in1        = (float *)malloc(nBytes);
        float *in2        = (float *)malloc(nBytes);
        float *correctOut = (float *)malloc(nBytes);
        float *outV1      = (float *)malloc(nBytes);
        float *outV2      = (float *)malloc(nBytes);

        // Khởi tạo mảng đầu vào với giá trị ngẫu nhiên trong đoạn [0, 1]
        for (size_t i = 0; i < N; ++i) {
            in1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            in2[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        // Cộng hai mảng trên host
        float hostTime     = addVec(in1, in2, correctOut, N, DeviceType::HOST);

        // Cộng hai mảng trên device theo version 1
        float deviceTimeV1 = addVec(in1, in2, outV1, N, DeviceType::DEVICE, KernelVersion::VERSION_1);
        
        // Cộng hai mảng trên device theo version 2
        float deviceTimeV2 = addVec(in1, in2, outV2, N, DeviceType::DEVICE, KernelVersion::VERSION_2);

        // Kiểm tra kết quả tính toán trên host và device (cả hai version)
        bool isCorrectV1 = true, isCorrectV2 = true;
        for (size_t i = 0; i < N; ++i) {
            if (correctOut[i] != outV1[i]) {
                isCorrectV1 = false;
                break;
            }
            if (correctOut[i] != outV2[i]) {
                isCorrectV2 = false;
                break;
            }
        }

        // Giải phóng bộ nhớ cho các mảng sau mỗi lần thử nghiệm
        free(in1);
        free(in2);
        free(correctOut);
        free(outV1);
        free(outV2);

        // Kiểm tra tính đúng đắn của kết quả
        if (!isCorrectV1 || !isCorrectV2) {
            fprintf(stderr, "[FAILED] Incorrect result!\n");
            exit(EXIT_FAILURE);
        } 
        else {
            printf("passed\n");
        }

        // Lưu kết quả thời gian chạy vào mảng chứa kết quả
        hostTimes[caseIdx]     = hostTime;
        deviceTimesV1[caseIdx] = deviceTimeV1;
        deviceTimesV2[caseIdx] = deviceTimeV2;
    }
    printf("‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n\n");


    // +========================================================+
    // |   Hiển thị bảng kết quả tổng hợp của các trường hợp    |
    // +--------------------------------------------------------+
    printf("[===== Result summary =====]\n");
    // In tiêu đề của bảng kết quả tổng hợp
    OUTER_LINE;
    printf("| %-12s| %-18s| %-18s| %-18s|\n", "Vector size", " Host time (ms)", "Device time (ms)", "Device time (ms)");
    printf("| %-12s| %-18s| %-18s| %-18s|\n", " ", " ", "   (Version 1)", "   (Version 2)");
    INNER_LINE;
    // In kết quả (thời gian chạy) của từng trường hợp thử nghiệm
    for (size_t caseIdx = 1; caseIdx < nTestCases; ++caseIdx) {
        printf("| %-12zu| %-18.3f| %-18.3f| %-18.3f|\n",
                VECTOR_SIZES[caseIdx], hostTimes[caseIdx], 
                deviceTimesV1[caseIdx], deviceTimesV2[caseIdx]);
    }
    // In dòng kẻ cuối cùng của bảng kết quả tổng hợp
    OUTER_LINE;

    // Giải phóng bộ nhớ cho các mảng chứa kết quả sau quá trình thử nghiệm
    free(hostTimes);
    free(deviceTimesV1);
    free(deviceTimesV2);

    return EXIT_SUCCESS;
}
