**Nhận xét**:

Dựa vào bảng tổng hợp thời gian chạy chương trình cộng hai vector trên host (CPU) và device (GPU), ta có một số nhận xét sau:

`1.` **Thời gian chạy trên CPU:**

- Khi kích thước vector tăng dần, thời gian chạy trên CPU tăng gần như tuyến tính. Điều này là hợp lý, vì CPU thực hiện cộng từng phần tử tuần tự, và càng nhiều phần tử thì thời gian xử lý càng lâu.

- Ở kích thước nhỏ (64, 256 phần tử), thời gian chạy rất ngắn (khoảng 0.01 ms). Tuy nhiên, khi kích thước tăng lên đến 16 triệu phần tử (16777216), thời gian chạy lên đến khoảng 80.0 - 120.0 ms.

`2.` **Thời gian chạy trên GPU (Version 1 và Version 2):**

- **Version 1** và **Version 2** cho kết quả khá tương đồng, với thời gian chạy thấp hơn đáng kể so với CPU, đặc biệt là khi kích thước vector lớn.

- Ở các kích thước nhỏ (64, 256, 1024 phần tử), thời gian chạy trên GPU hơi cao hơn so với CPU nhưng không đáng kể.

- Tuy nhiên, khi kích thước vector tăng lên (bắt đầu từ 4096 phần tử), GPU bắt đầu vượt trội so với CPU. Với các kích thước lớn như 16 triệu phần tử, GPU chỉ mất khoảng 0.8 - 1.0 ms, nhanh hơn CPU khoảng 100 lần.

**Kết luận**:

- Khi kích thước vector lớn hơn, thời gian chạy trên GPU có xu hướng tăng nhẹ, nhưng vẫn giữ ở mức rất thấp so với CPU. Điều này cho thấy GPU có thể xử lý khối lượng tính toán song song hiệu quả hơn rất nhiều so với CPU khi xử lý các vector lớn.

- GPU cho thấy hiệu suất vượt trội khi xử lý các vector lớn, nhưng với các vector nhỏ, chi phí truyền dữ liệu có thể làm giảm hiệu suất tổng thể.

- GPU đạt hiệu suất cao nhất khi kích thước vector lớn, nhờ vào khả năng tính toán song song trên nhiều thread.
