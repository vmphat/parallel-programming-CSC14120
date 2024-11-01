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

# Parallel Execution in CUDA (Part 1)

## Ôn tập Quiz 2

- Với mảng 3D, ta cắt lát các mặt phẳng và xếp theo thứ tự, mỗi mặt phẳng sẽ sắp xếp theo row-/column-major => `height * width * z + ...`

## GPU compute architecture

- Tốc độ truy xuất L1 Cache (của 1 SM) > Global Memory
- Warp -> SIMT: Nhiều threads chạy cùng 1 câu lệnh (trên dữ liệu riêng của mỗi thread)
- Warp divergence: hiện tượng rẽ nhánh theo threadIdx
- Đoạn inactive: thực ra all threads cùng chạy nhưng nó chỉ tính chứ không ghi kết quả
  - Ý tưởng: Tránh việc chạy đan xen

```Bài tập
| BlockSize | Num Blocks |  Num Threads | Occupancy |
| --- | --- | --- | --- |
| 32 | 32 | 1024 | 50% |
| 768 | 2 | 1536 | 75% |
```

## Note

- Tuần sau học về bài toán tối ưu cụ thể
- Làm bài quiz (trong ngày hôm nay)

# Parallel Execution in CUDA (Part 2)

- Ôn tập Quiz 3

## Overview

- Sau này project cũng sẽ trải qua các giai đoạn:
  1. Phân tích task
  2. Cài tuần tự
  3. Cài song song
     - Song song version 1
     - Song song version 2
     - Song song version 3
     - ...
- Stride: bước nhảy
- Tài liệu tham khảo các hàm atomic trong CUDA: **CUDA C++ Programming Guide**
  => 1 report do NVIDIA cung cấp
- Ý tưởng tráng phân kỳ là dồn các thread chạy về 1 bên, thread không chạy về 1 bên
  - Tuy nhiên, số lượng thread cần chạy là không đổi

## Note

- Cài đặt ý tưởng kernel version 3 vào lab 2
- Cài đặt nhân ma trận bằng CUDA trong lab
  - Dùng 1 vòng lặp bên trong cùng

```
- Câu 1:
  - Cài 3 hàm kernel để thực hiện reduction
  - Cần cài các bước cấp phát, copy, gọi hàm, free
  - Chạy trong file notebook, tìm cách lý giải kết quả, nhận xét
  - Dùng `nvprof` để xem thông tin về quá trình thực thi để có chiến lược cài đặt hiệu quả hơn
    - VD: Thời gian chạy kernel là rất ít so với thời gian copy dữ liệu qua lại
    - Ta sẽ nhận xét từ bảng thông báo này, chỉ cần để ý **GPU activites** thôi
  - Chạy với blocksize khác nhau rồi điền bảng + nhận xét
- Câu 2:
  - Cài đặt bài toán nhân 2 ma trận
  - Cài 2 hàm kernel:
    - Version 1: Đơn giản, dùng global memory
    - Version 2: Dùng shared memory
  - Dùng `nvprof` để xem chi tiết thực thi và nhận xét
  - Phải trả lời thêm câu hỏi trên notebook
```

- Lab làm trong 2 tuần
- Tuần này không có Quiz
- Đề đồ án:
  - Cài đặt mạng Neuron (bình thường) cho bài toán **Fashion MNIST** bằng C và song song hóa bằng CUDA
  - Bản chất là nhân ma trận
  - Nội dung chi tiết thầy sẽ gửi qua document sau
  - Có thể tìm hiểu cách cài mạng Neuron bằng C trước
  - Làm đồ án theo nhóm 2 hoặc 3
