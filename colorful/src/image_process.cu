#include "image_process.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <iostream>
#include <stdio.h>

__global__ void point_index_2_image_index(float *bev_points, int *point_size,
                                     int *width, int *height, float *scale_x, float *scale_y,
                                     uint64_t *points_index2bev_height,
                                     int *points_index) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
    float point_x = bev_points[3 * idx];
    float point_y = bev_points[3 * idx + 1];
    float point_z = bev_points[3 * idx + 2];
    int x = static_cast<int>(point_x * (*scale_x));
    int y = static_cast<int>(point_y * (*scale_y));
    uint64_t pixel_index = static_cast<uint64_t>(y * *width + x);
    pixel_index <<= 28;
    pixel_index |= *((uint32_t*)&(point_z));
    points_index2bev_height[idx] = pixel_index;
    points_index[idx] = idx;
}

__global__ void identify_pixel_ranges(uint64_t* pixel_index, uint2* ranges, int *point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= *point_size)
		return;
    uint32_t pixel = pixel_index[idx] >> 28;
	if (idx == 0) {
		ranges[pixel].x = 0;
    }
	else {
		uint32_t pre_pixel = pixel_index[idx - 1] >> 28;
		if (pixel != pre_pixel) {
			ranges[pre_pixel].y = idx;
			ranges[pixel].x = idx;
		}
	}
	if (idx == *point_size - 1) {
		ranges[pixel].y = *point_size;
    }
}

__global__ void init_ranges(uint2* ranges, int *width, int *height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= *width * *height) {
        return;
    }
    ranges[idx].x = 0;
    ranges[idx].y = 0;
}

__global__ void replenish(float* __restrict__ bev_height, const bool* const __restrict__ bev_mask,
                        int* __restrict__ width, int* __restrict__ height,
                        bool* __restrict__ bev_mask_new, int radius_max, int count_max) {
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y;
    int pixel_index = pixel_x + pixel_y * *width;
    if (pixel_index < *width * *height && bev_mask[pixel_index]) {
        return;
    }
    int test_count = 0;
    for (int radius = radius_max / 2; radius < radius_max + 2; radius += radius_max / 2) {
        int start_x = pixel_x - radius;
        int start_y = pixel_y - radius;
        int end_x = pixel_x + radius;
        int end_y = pixel_y + radius;
        for (int x = start_x; x <= end_x; x++) {
            int y = start_y;
            if (y < 0 || y >= *height) break;
            if (x < 0 || x >= *width) continue;
            if (x < *width && y < *height && bev_mask[x + y * *width] == true) {
                test_count++;
            }
        }
        for (int x = start_x; x <= end_x; x++) {
            int y = end_y;
            if (y < 0 || y >= *height) break;
            if (x < 0 || x >= *width) continue;
            if (x < *width && y < *height && bev_mask[x + y * *width] == true) {
                test_count++;
            }
        }
        for (int y = start_y + 1; y < end_y; y++) {
            int x = start_x;
            if (x < 0 || x >= *width) break;
            if (y < 0 || y >= *height) continue;
            if (bev_mask[x + y * *width]) {
                test_count++;
            }
        }
        for (int y = start_y + 1; y < end_y; y++) {
            int x = end_x;
            if (x < 0 || x >= *width) break;
            if (y < 0 || y >= *height) continue;
            if (bev_mask[x + y * *width]) {
                test_count++;
            }
        }
    }

    if (test_count == 0) {
        return;
    }
    int count = 0;
    float height_sum = 0.0;
    for (int radius = 1; radius < radius_max && count < count_max; radius++) {
        int start_x = pixel_x - radius;
        int start_y = pixel_y - radius;
        int end_x = pixel_x + radius;
        int end_y = pixel_y + radius;
        int tmp_count = 0;
        for (int x = start_x; x <= end_x; x++) {
            int y = start_y;
            if (y < 0 || y >= *height) break;
            if (x < 0 || x >= *width) continue;
            if (x < *width && y < *height && bev_mask[x + y * *width] == true) {
                tmp_count++;
                height_sum += bev_height[x + y * *width];
            }
        }
        for (int x = start_x; x <= end_x; x++) {
            int y = end_y;
            if (y < 0 || y >= *height) break;
            if (x < 0 || x >= *width) continue;
            if (x < *width && y < *height && bev_mask[x + y * *width] == true) {
                tmp_count++;
                height_sum += bev_height[x + y * *width];
            }
        }
        for (int y = start_y + 1; y < end_y; y++) {
            int x = start_x;
            if (x < 0 || x >= *width) break;
            if (y < 0 || y >= *height) continue;
            if (bev_mask[x + y * *width]) {
                tmp_count++;
                height_sum += bev_height[x + y * *width];
            }
        }
        for (int y = start_y + 1; y < end_y; y++) {
            int x = end_x;
            if (x < 0 || x >= *width) break;
            if (y < 0 || y >= *height) continue;
            if (bev_mask[x + y * *width]) {
                tmp_count++;
                height_sum += bev_height[x + y * *width];
            }
        }
        count += tmp_count;
    }
    if (count > 0) {
        bev_height[pixel_index] = height_sum / float(count);
        bev_mask_new[pixel_index] = true;
    }
}

__device__ void insertionSortPairs(float* keys, float* values, int n) {
    for (int i = 1; i < n; i++) {
        float key = keys[i];
        float value = values[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > key) {
            keys[j + 1] = keys[j];
            values[j + 1] = values[j];
            j--;
        }
        keys[j + 1] = key;
        values[j + 1] = value;
    }
}

__device__ void heapifyDown(float minHeap[], float keyArray[], int idx, int size) {
    int smallest = idx;
    while (true) {
        int leftChild = 2 * smallest + 1;
        int rightChild = 2 * smallest + 2;
        
        if (leftChild < size && minHeap[leftChild] < minHeap[smallest]) {
            smallest = leftChild;
        }
        if (rightChild < size && minHeap[rightChild] < minHeap[smallest]) {
            smallest = rightChild;
        }

        if (smallest == idx) {
            break;
        }

        float tmp = minHeap[idx];
        minHeap[idx] = minHeap[smallest];
        minHeap[smallest] = tmp;
        
        tmp = keyArray[idx];
        keyArray[idx] = keyArray[smallest];
        keyArray[smallest] = tmp;

        idx = smallest;
    }
}

__device__ void heapifyUp(float minHeap[], float keyArray[], int idx) {
    while (idx > 0) {
        int parentIdx = (idx - 1) / 2;
        if (minHeap[idx] < minHeap[parentIdx]) {
            float tmp = minHeap[idx];
            minHeap[idx] = minHeap[parentIdx];
            minHeap[parentIdx] = tmp;
            tmp = keyArray[idx];
            keyArray[idx] = keyArray[parentIdx];
            keyArray[parentIdx] = tmp;
            idx = parentIdx;
        } else {
            break;
        }
    }
}

__device__ int distanceToLine(int x1, int y1, int x2, int y2, int x3, int y3) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    int lineLengthSquared = dx * dx + dy * dy;
    int num = std::abs(dy * x3 - dx * y3 + x2 * y1 - y2 * x1);
    int denom = static_cast<int>(sqrtf(static_cast<float>(lineLengthSquared)));
    int distance = num / denom;
    return distance;
}

__global__ void render(float *bev_height, bool *bev_mask,
                       int *width, int *height,
                       float *bev_points,
                       uint2 *ranges,
                       int *points_index,
                       int *uvs, int uv_length) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= *width || pixel_y >= *height) {
        return;
    }
    int pixel_index = pixel_x + pixel_y * *width;
    uint64_t min_distance = 100000000;
    uint64_t second_distance = 100000000;
    int min_index = 1;
    int second_index = 0;
    for (int i = 0; i < uv_length; i++) {
        uint64_t distanse = static_cast<uint64_t>(uvs[2 * i] - pixel_x) * static_cast<uint64_t>(uvs[2 * i] - pixel_x) +
                            static_cast<uint64_t>(uvs[2 * i + 1] - pixel_y) * static_cast<uint64_t>(uvs[2 * i + 1] - pixel_y);
        if (distanse < min_distance) {
            second_distance = min_distance;
            second_index = min_index;
            min_distance = distanse;
            min_index = i;
        }
        else if (distanse < second_distance) {
            second_distance = distanse;
            second_index = i;
        }
    }
    int distance = 10;
    if (uv_length > 1) {
        distance = distanceToLine(uvs[2 * min_index], uvs[2 * min_index + 1],
                                  uvs[2 * second_index], uvs[2 * second_index + 1],
                                  pixel_x, pixel_y);
    }
    int kernel_size = 10;
    if (distance > 150) {
        kernel_size = 10;
    }
    else {
        kernel_size = 4;
    }
    __shared__ float depth_box[21];
    __shared__ float depth_weight[21];
    for (int i = 0; i < 20; i++) {
        depth_weight[i] = 0.0;
        depth_box[i] = -200;
    }
    int size = 0;
    for (int i = -kernel_size; i <= kernel_size; i++) {
        for (int j = -kernel_size; j <= kernel_size; j++) {
            if (pixel_y + i < 0 || pixel_y + i >= *height || 
                pixel_x + j < 0 || pixel_x + j >= *width) {
                continue;
            }
            uint2 range = ranges[pixel_x + j + (pixel_y + i) * *width];
            if (range.y > range.x) {
                bev_mask[pixel_index] = true;
                if (size < 20) {
                    depth_box[size] = bev_points[3 * points_index[range.y - 1] + 2];
                    depth_weight[size] = exp(-0.005 * (i * i + j * j));
                    heapifyUp(depth_box, depth_weight, size);
                    size++;
                } else if (bev_points[3 * points_index[range.y - 1] + 2] > depth_box[0]){
                    depth_box[0] = bev_points[3 * points_index[range.y - 1] + 2];
                    depth_weight[0] = exp(-0.002 * (i * i + j * j));
                    heapifyDown(depth_box, depth_weight, 0, size);
                }
            }
        }
    }
    if (size < 2) {
        bev_mask[pixel_index] = false;
    }
    else {
        float res = 0.0;
        float weight_sum = 0.0;
        // TODO: pop front

        // if (size == 1) {
        //     bev_mask[pixel_index] = true;
        //     bev_height[pixel_index] = depth_box[0];
        // }
        // else {
        int halfSize = size / 2;
        for (int i = 0; i < halfSize; ++i) {
            depth_box[0] = depth_box[size - 1];
            depth_weight[0] = depth_weight[size - 1];
            size--;
            heapifyDown(depth_box, depth_weight, 0, size);
        }
        for (int i = 0; i < size; i++) {
            res += depth_box[i] * depth_weight[i];
            weight_sum += depth_weight[i];
        }
        bev_mask[pixel_index] = true;
        bev_height[pixel_index] = res / weight_sum;
        // }
    }
    __syncthreads();
}

__global__ void sync_bev_mask(bool *bev_mask, bool *bev_mask_new, int *height, int *width) {
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y;
    if (pixel_x >= *width || pixel_y >= *height) {
        return;
    }
    int pixel_index = pixel_x + pixel_y * *width;
    if (bev_mask_new[pixel_index]) {
        bev_mask[pixel_index] = true;
    }
}

__global__ void test(uint64_t* pixel_index, int *points_index) {
    for (int i = 0; i < 100; i++) {
        // printf("%d ", (pixel_index[i] >> 28));
    }
    // for (int i = 0; i < 100; i++) {
    //     printf("%d ", points_index[i]);
    // }
}

void ImageProcess::DepthImage(float* h_bev_points, int* h_point_size,
                    int* h_width, int* h_height, float* h_scale_x, float* h_scale_y,
                    float* h_bev_height, bool* h_bev_mask, int* h_uvs, int uv_length) {
    float *bev_points;
    int *width, *height, *point_size;
    float *scale_x, *scale_y;
    uint64_t *pixel_index, *pixel_index_sorted;
    int *points_index, *points_index_sorted;
    uint2* ranges;
    float *bev_height;
    bool *bev_mask;
    bool *bev_mask_new;
    int *uvs;
    cudaMalloc((void**)&bev_points, 3 * (*h_point_size) * sizeof(float));
    cudaMalloc((void**)&scale_x, sizeof(float));
    cudaMalloc((void**)&scale_y, sizeof(float));
    cudaMalloc((void**)&point_size, sizeof(int));
    cudaMalloc((void**)&width, sizeof(int));
    cudaMalloc((void**)&height, sizeof(int));
    cudaMalloc((void**)&bev_height, (*h_width) * (*h_height) * sizeof(float));
    cudaMalloc((void**)&pixel_index, (*h_point_size) * sizeof(uint64_t));
    cudaMalloc((void**)&points_index, (*h_point_size) * sizeof(int));
    cudaMalloc((void**)&pixel_index_sorted, (*h_point_size) * sizeof(uint64_t));
    cudaMalloc((void**)&points_index_sorted, (*h_point_size) * sizeof(int));
    cudaMalloc((void**)&bev_mask, (*h_width) * (*h_height) * sizeof(bool));
    cudaMalloc((void**)&ranges, (*h_width) * (*h_height) * sizeof(uint2));
    cudaMalloc((void**)&bev_mask_new, (*h_width) * (*h_height) * sizeof(bool));
    cudaMalloc((void**)&uvs, 2 * uv_length * sizeof(int));

    cudaMemcpy(bev_points, h_bev_points, 3 * (*h_point_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_x, h_scale_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_y, h_scale_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(point_size, h_point_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(width, h_width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(height, h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bev_height, h_bev_height, (*h_height * *h_width) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bev_mask, h_bev_mask, (*h_height * *h_width) * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(uvs, h_uvs, 2 * uv_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(bev_mask_new, 0, (*h_width) * (*h_height) * sizeof(bool));
    point_index_2_image_index<<< (*h_point_size + 255) / 256, 256 >>>(bev_points, point_size, width, height, scale_x, scale_y, pixel_index, points_index);
    // test<<< 1, 1 >>>(pixel_index, points_index);
    cudaDeviceSynchronize();
    void *sort_storage_tmp = nullptr;
    size_t sort_storage_size = 0;
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixel_index, pixel_index_sorted, points_index, points_index_sorted, *h_point_size);
    cudaMalloc(&sort_storage_tmp, sort_storage_size);
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixel_index, pixel_index_sorted, points_index, points_index_sorted, *h_point_size);
    cudaDeviceSynchronize();
    // test<<< 1, 1 >>>(pixel_index_sorted, points_index_sorted);
    cudaFree(sort_storage_tmp);
    init_ranges<<< (*h_width * *h_height + 255) / 256 , 256>>>(ranges, width, height);
    identify_pixel_ranges<<< (*h_point_size + 255) / 256, 256 >>> (pixel_index_sorted, ranges, point_size);
    dim3 grid_dim(*h_width, *h_height);
    render<<< grid_dim, 1 >>>(bev_height, bev_mask, width, height,
                              bev_points, ranges, points_index_sorted,
                              uvs, uv_length);
    cudaDeviceSynchronize();
    for (int i = 0; i < 100; i++) {
        std::cout << "\rProgress: " << i + 1 << "/" << 100
          << " [" << std::fixed
          << (static_cast<double>(i + 1) / 100) * 100 << "%]";
        std::cout.flush();
        replenish<<< grid_dim, 1 >>>(bev_height, bev_mask, width, height, bev_mask_new, 70, 6);
        sync_bev_mask<<< grid_dim, 1 >>>(bev_mask, bev_mask_new, height, width);
        cudaMemset(bev_mask_new, 0, (*h_width) * (*h_height) * sizeof(bool));
        cudaDeviceSynchronize();
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();

    cudaMemcpy(h_bev_height, bev_height, (*h_height * *h_width) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bev_mask, bev_mask, (*h_height * *h_width) * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(bev_points);
    cudaFree(scale_x);
    cudaFree(scale_y);
    cudaFree(point_size);
    cudaFree(width);
    cudaFree(height);
    cudaFree(bev_height);
    cudaFree(pixel_index);
    cudaFree(pixel_index_sorted);
    cudaFree(bev_mask);
    cudaFree(ranges);
    cudaFree(points_index_sorted);
    cudaFree(points_index);
    cudaFree(bev_mask_new);
}
