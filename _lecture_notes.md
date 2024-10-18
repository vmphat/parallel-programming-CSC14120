# Parallel Programming

# Course Introduction

- ILP: số lượng câu lệnh xử lý song song
- Tensor: 1 ma trận nhiều chiều
  => Thường dùng cho train deep learning
- GPU có mạnh hơn CPU về FLOPS
- Cache nằm trong CPU nên tốc độ truy xuất nhanh hơn so với RAM
- Trong GPU:
  - CU, Cache thường ít
  - Có nhiều ALU nhưng kích thước nhỏ hơn (yếu hơn)
- Nhìn chung:
  - Tính toán phức tạp, cần tính nhanh
    => dùng CPU
  - Tính toán đơn giản, cần nhiều phép tính
    => dùng GPU

## Course assessment

- **Small quiz** (_làm trong ngày_): 10%
  - Trắc nghiệm + Câu hỏi nhỏ
  - Khoảng 5-6 bài
- **Lab exercises**: 40%
  - Thường là bổ sung code vào sườn được cung cấp
  - Khoảng 5-6 bài
- **Group final project**: 50%
  - Nhóm 2-3 sinh viên
  - Đề bài: Song song hóa một bài toán lớn
- Chấm bài trên môi trường Colab

# Introduction to CUDA C/C++ (Part I)

- Mô hình lập trình CUDA C/C++

## Data parallelism (Song song hóa theo dữ liệu)

- Vd: việc chuyển grayscale ảnh, việc biến đổi mỗi pixel là độc lập với nhau
- Có 2 hướng thực hiện song song

  - **Song song theo tác vụ** (task parallelism): Có nhiều tác vụ được thực hiện song song
    vd: vừa nghe nhạc, vừa coding
  - **Song song theo dữ liệu** (data parallelism): thực hiện cùng một công việc trên nhiều dữ liệu khác nhau
    vd: 2 người cùng giặt đồ, mỗi người giặt 1 cái áo

- Kiến trúc SIMD: Single Instruction, Multiple Data
  - Some instructions execute concurrently
- Kiến trúc SPMD: Single program, multiple data
  - Code is identical across all threads
  - Execution path may differ
    => Các thread chạy cùng 1 chương trình, nhưng đường thực thi có thể khác nhau
    => CUDA đi theo hướng này

## CUDA Execution Model

- Heterogeneous host (CPU) + device (GPU) application C progam
  - Serial parts run in **host** C code
  - Parallel parts run in **device** SPMD kernel code
    - All threads in grid run same kernel code
    - Each thread has index used to compute memory address

## Adding 2 vectors

- Với bài toán song song, xác định xem phần nào song song được, công việc nào độc lập với công việc nào
  => Chỉ code song song cho những phần có thể song song
- Chương trình song song chạy trên 1 grid:

  - 1 grid chia thành nhiều block
  - 1 block chứa nhiều threads
  - Số lượng thread trên mỗi block là đồng nhất

- cudaMemcpy: là **synchronous** (đồng bộ hóa)
  - Chờ hàm này chạy xong thì chương trình mới chạy câu lệnh tiếp theo

## Bài tập tuần này

- 1 bài quiz:
  - Trắc nghiệm về nội dung bài học
  - Deadline: 1 ngày
- 1 bài lab:
  - Đề thầy đăng lên sau
  - Nộp file notebook thầy cung cấp sẵn
  - Nộp source code và nộp file notebook
  - Deadline: 1 tuần

# Introduction to CUDA C/C++ (Part II)

- Ôn tập lại quiz 1
- Ôn tập lại HW0
  - gridSize = (n/2-1)/blockDim.x + 1

## Adding 2 matrices

- Chiều 1,2,3 tương ứng: x,y,z
  - Nếu z=1 thì là 2 chiều
  - Trục column: x
  - Trục row: y
    => Luôn chọn kiểu này để tối ưu hiệu năng CUDA

## RGB to Grayscale

- Công thức cộng ba màu chia trung bình cũng được nhưng hình ảnh sẽ không sắc nét
  => Ý tưởng: Màu nào dễ nhận diện sẽ có trọng số cao hơn (green >> red >> blue)
- Input: mỗi phần tử là bộ ba số (r,g,b)
- Output: mỗi phần tử là 1 con số

## Blurring image

- Làm mờ có khả năng lọc nhiễu cho ảnh, đặc biệt là nhiễu li ti
- Làm mờ theo từng kênh màu

## Note

- Làm bài Lab (HW1)

```
- Câu 1: RGB-2-Grayscale
  - Đọc file .pnm (text file):
    - R,G,B là 3 con số liên tiếp nhau trong mỗi dòng
  - Điền chỗ trống trong file .cu
  - Phải chú ý kiểu dữ liệu của biến
  - Phải viết hàm kiểm lỗi khi chạy hàm kernel (Slide 01 - trang 20)
  - So sánh kết quả host vs. device dựa trên độ lệch
    - Nếu độ lỗi nhỏ thì code đúng
  - Báo cáo: Chạy với các kích thước khác nhau và thể hiện chương trình có thể bắt lỗi

- Câu 2:
  - Có khung rồi
  - Bài toán tích chập
    - Cơ bản là tổng có trọng số
    - Tổng của các trọng số thường bằng 1 (không phải luôn đúng)
  - Cần cài với kích thước filter linh động (3,5,7,v.v.)
  - Nếu áp dụng lên ảnh RGB thì phải áp lên từng kênh màu
  - Chú ý việc xử lý vùng biên của ảnh:
    1. Không làm gì hết => Giảm chiều bức ảnh
    2. Duplicate phần biên => Lab làm theo kiểu này
  - Phải chú ý kiểu dữ liệu của biến
  - Host phải dùng 4 vòng for:
    - 2 vòng ngoài là duyệt từng pixel
  - Kernel: dùng 2 vòng lặp trong
```

- Làm bài quiz (trong ngày hôm nay)
