# Nhận xét về sự thay đổi trong thời gian thực thi khi thay đổi kích thước block trong chương trình làm mờ ảnh RGB

Khi thay đổi kích thước block trong chương trình làm mờ ảnh RGB, ta có thể nhận thấy sự thay đổi trong thời gian thực thi như sau:

1. **Block 16x16**:
	- Thời gian thực thi có thể tương đối dài do kích thước block nhỏ, dẫn đến việc cần nhiều block hơn để xử lý toàn bộ ảnh.
	- Điều này có thể gây ra overhead do việc quản lý nhiều block và thread hơn.

2. **Block 16x32**:
	- Thời gian thực thi có thể giảm so với block 16x16 do kích thước block lớn hơn, giảm số lượng block cần thiết.
	- Tuy nhiên, nếu kích thước block không phù hợp với kích thước ảnh, có thể dẫn đến việc không sử dụng hết tài nguyên của GPU.

3. **Block 16x64**:
	- Thời gian thực thi có thể tiếp tục giảm nếu kích thước block phù hợp với kích thước ảnh và tài nguyên của GPU.
	- Tuy nhiên, nếu block quá lớn, có thể dẫn đến việc không sử dụng hết các thread trong mỗi block, gây lãng phí tài nguyên.

4. **Block 32x32**:
	- Thời gian thực thi có thể tối ưu nhất nếu kích thước block phù hợp với kích thước ảnh và tài nguyên của GPU.
	- Kích thước block 32x32 thường được sử dụng phổ biến do phù hợp với kiến trúc của nhiều GPU, giúp tận dụng tối đa tài nguyên.

**Nhận xét chung**:
- Kích thước block ảnh hưởng trực tiếp đến thời gian thực thi của chương trình. Kích thước block quá nhỏ hoặc quá lớn đều có thể dẫn đến việc không sử dụng hiệu quả tài nguyên của GPU.
- Việc chọn kích thước block phù hợp là rất quan trọng để tối ưu hóa thời gian thực thi của chương trình. Thông thường, kích thước block 32x32 là lựa chọn tốt cho nhiều ứng dụng do phù hợp với kiến trúc của nhiều loại GPU.

```C++
#define FILTER_WIDTH 9
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
        uchar3 * outPixels,
        bool useDevice=false, dim3 blockSize=dim3(1, 1), int kernelType=1)
{
	if (useDevice == false)
	{
		//TODO

		/**
		 * Blur the input (RGB) image using CPU
		**/

		// Loop through each pixel in the image
		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				// Calculate the filter radius
				int filterRadius = filterWidth / 2;
				// Initialize the weighted sum for each color channel of the pixel
				float3 sum = make_float3(0.0f, 0.0f, 0.0f);

				// Loop over the filter
				for (int filterR = 0; filterR < filterWidth; ++filterR)
				{
					for (int filterC = 0; filterC < filterWidth; ++filterC)
					{
						// Compute the position of the neighbor pixel in the input image
						int imageR = r + filterR - filterRadius;
						int imageC = c + filterC - filterRadius;
						// Clamp to boundary of the image
						imageR = min(max(imageR, 0), height - 1);
						imageC = min(max(imageC, 0), width  - 1);

						// Get the pixel value at the image position
						uchar3 pixel = inPixels[imageR * width + imageC];
						// Get the filter value at the filter position
						float filterValue = filter[filterR * filterWidth + filterC];
						
						// Accumulate the weighted sum for each color channel
						sum.x += pixel.x * filterValue;
						sum.y += pixel.y * filterValue;
						sum.z += pixel.z * filterValue;
					}
				}

				// Assign the blurred pixel to the output image
				outPixels[r * width + c] = make_uchar3(sum.x, sum.y, sum.z);
			}
		}
	}
	else // Use device
	{
	}
}
```


`3.` **Thời gian chạy với các kích thước block khác nhau:**

- **Kích thước block nhỏ (16x16):**
  - Số lượng thread trên mỗi block nhỏ, dẫn đến cần nhiều block để bao phủ toàn bộ ảnh. Số lượng block tăng làm tăng overhead về quản lý block và giao tiếp giữa các SM.
  - Hơn nữa, mỗi block chỉ có một lượng nhỏ SMEM, nên không tận dụng hết băng thông của shared memory.

- **Kích thước block lớn hơn (16x32, 16x64, 32x32):**
  - Khi block lớn hơn, số lượng thread trên mỗi block tăng, giúp giảm số lượng block cần thiết, giảm overhead quản lý.
  - Với kích thước block 32x32, số lượng thread trong block gần với giới hạn tối đa của một SM (1024 thread). Điều này giúp tận dụng tối đa tài nguyên trên mỗi SM, dẫn đến thời gian chạy tối ưu.

- **Sự khác biệt nhỏ giữa các kích thước block lớn hơn (16x32, 16x64, 32x32):**
  - Dù kích thước block tăng, thời gian chạy chỉ thay đổi nhẹ do ảnh hưởng từ cách các thread được lập lịch và sử dụng tài nguyên SMEM và registers. Với kích thước block 32x32, số lượng thread vừa phải, tối ưu được hiệu suất SM.

---

### **Tóm tắt:**
1. **SMEM** cải thiện hiệu suất vì giảm số lần truy cập GMEM (thời gian chạy của kernel 2 nhanh hơn kernel 1).
2. **CMEM** cải thiện hiệu suất hơn nữa khi dữ liệu bất biến như filter được sử dụng (kernel 3 nhanh hơn kernel 2).
3. **Kích thước block ảnh hưởng thời gian chạy**:
   - Block nhỏ có nhiều overhead hơn.
   - Block lớn tận dụng tốt hơn tài nguyên SM nhưng phải cân đối để không vượt quá giới hạn tài nguyên (threads, SMEM, registers). 
   - Kích thước block 32x32 tối ưu cho ảnh kích thước 512x512 trên GPU Tesla T4.
