#include "distribute_project.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

__global__ void project_cuda(double* rotation, double* translation,
                             unsigned char* image, unsigned char* image_mask,
                             unsigned char* image_seg,
                             int *height, int *width,
                             double* K,
                             float *points,
                             float* rgbs,
                             float* rgb_counts,
                             int* label_map,
                             int** label_counts,
                             int* point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
	float3 p_m = { points[3 * idx], points[3 * idx + 1], points[3 * idx + 2] };
    // to calculate p_c
    float3 p_c = {0, 0, 0};
    p_c.x = rotation[0] * p_m.x + rotation[1] * p_m.y + rotation[2] * p_m.z + translation[0];
    p_c.y = rotation[3] * p_m.x + rotation[4] * p_m.y + rotation[5] * p_m.z + translation[1];
    p_c.z = rotation[6] * p_m.x + rotation[7] * p_m.y + rotation[8] * p_m.z + translation[2];
    if (p_c.z > 18 || p_c.z < 1 ||
        p_c.x < -18 || p_c.x > 18) {
        return;
    }
    float3 p_cuni = {p_c.x / p_c.z, p_c.y / p_c.z, 1.0};
    float u = K[0] * p_cuni.x + K[1] * p_cuni.y + K[2];
    float v = K[3] * p_cuni.x + K[4] * p_cuni.y + K[5];
    int x = static_cast<int>(u);
    int y = static_cast<int>(v);
    if (x < 0 || x >= *width || y < 0 || y >= *height) {
        return;
    }
    int pixel_index = y * *width + x;
    if (image_mask[pixel_index] == 0) {
        return;
    }
    unsigned char b = image[3 * pixel_index];
    unsigned char g = image[3 * pixel_index + 1];
    unsigned char r = image[3 * pixel_index + 2];
    rgbs[3 * idx] += r;
    rgbs[3 * idx + 1] += g;
    rgbs[3 * idx + 2] += b;
    rgb_counts[idx] += 1;
    label_counts[label_map[static_cast<int>(image_seg[pixel_index])]][idx] += 1;
}

__global__ void project_cuda_nearest(double* rotation, double* translation,
                             unsigned char* image, unsigned char* image_mask,
                             unsigned char* image_seg,
                             int *height, int *width,
                             double* K,
                             float *points,
                             float* rgbs,
                             float* rgb_counts,
                             int* nearest_label,
                             float *nearest_z,
                             int* point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
	float3 p_m = { points[3 * idx], points[3 * idx + 1], points[3 * idx + 2] };
    // to calculate p_c
    float3 p_c = {0, 0, 0};
    p_c.x = rotation[0] * p_m.x + rotation[1] * p_m.y + rotation[2] * p_m.z + translation[0];
    p_c.y = rotation[3] * p_m.x + rotation[4] * p_m.y + rotation[5] * p_m.z + translation[1];
    p_c.z = rotation[6] * p_m.x + rotation[7] * p_m.y + rotation[8] * p_m.z + translation[2];
    if (p_c.z > 100 || p_c.z < 8 ||
        p_c.x < -30 || p_c.x > 30) {
        return;
    }

    // float distance = sqrtf(p_c.x * p_c.x + p_c.y * p_c.y + p_c.z * p_c.z);
    float3 p_cuni = {p_c.x / p_c.z, p_c.y / p_c.z, 1.0};
    float u = K[0] * p_cuni.x + K[1] * p_cuni.y + K[2];
    float v = K[3] * p_cuni.x + K[4] * p_cuni.y + K[5];
    int x = static_cast<int>(u);
    int y = static_cast<int>(v);
    if (x < 0 || x >= *width || y < 0 || y >= *height) {
        return;
    }
    int pixel_index = y * *width + x;
    if (image_mask[pixel_index] == 0) {
        return;
    }
    unsigned char b = image[3 * pixel_index];
    unsigned char g = image[3 * pixel_index + 1];
    unsigned char r = image[3 * pixel_index + 2];
    // float weight = 1.5;
    // if (distance > 18) {
    //     weight = 0.1;
    // }
    // rgbs[3 * idx] += r * weight;
    // rgbs[3 * idx + 1] += g * weight;
    // rgbs[3 * idx + 2] += b * weight;
    // rgb_counts[idx] += weight;

    if(p_c.z < nearest_z[idx]){
        nearest_label[idx] = static_cast<int>(image_seg[pixel_index]) % 67;
        nearest_z[idx] = p_c.z;
        rgbs[3 * idx] = r;
        rgbs[3 * idx + 1] = g;
        rgbs[3 * idx + 2] = b;
        rgb_counts[idx] = 1;
    }
}

__global__ void mix_color_and_label_cuda(float *points, float* rgbs, int* labels, float* rgb_counts, int** label_counts, int* point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
    if (rgb_counts[idx] > 0) {
        rgbs[3 * idx] /= rgb_counts[idx];
        rgbs[3 * idx + 1] /= rgb_counts[idx];
        rgbs[3 * idx + 2] /= rgb_counts[idx];
    }
    int max_label = 0;
    int max_count = 0;
    for (int i = 0; i < 6; i++) {
        if (label_counts[i][idx] > max_count) {
            max_count = label_counts[i][idx];
            max_label = i;
        }
    }
    labels[idx] = max_label;
}

__global__ void mix_color_cuda_nearest(float *points, float* rgbs, float* rgb_counts, int* point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
    if (rgb_counts[idx] > 0) {
        rgbs[3 * idx] /= rgb_counts[idx];
        rgbs[3 * idx + 1] /= rgb_counts[idx];
        rgbs[3 * idx + 2] /= rgb_counts[idx];
    }
}

__global__ void transform_points_to_m(float* points_m, float* points_g, double* rotation, double* translation, int* point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
	float3 p_g = { points_g[3 * idx], points_g[3 * idx + 1], points_g[3 * idx + 2] };
    points_m[3 * idx] = rotation[0] * p_g.x + rotation[1] * p_g.y + rotation[2] * p_g.z + translation[0];
    points_m[3 * idx + 1] = rotation[3] * p_g.x + rotation[4] * p_g.y + rotation[5] * p_g.z + translation[1];
    points_m[3 * idx + 2] = rotation[6] * p_g.x + rotation[7] * p_g.y + rotation[8] * p_g.z + translation[2];
}

__global__ void point_pixel_index(float *points, int *point_size,
                            int *width, int *height, float *scale_x, float *scale_y,
                            int *pixels_index,
                            int *points_index) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
    float point_x = points[3 * idx];
    float point_y = points[3 * idx + 1];
    float x = point_x * (*scale_x);
    float y = point_y * (*scale_y);
    int pixel_index = static_cast<int>(y) * *width + static_cast<int>(x);
    pixels_index[idx] = pixel_index;
    points_index[idx] = idx;
}

__global__ void point_pixel_index_nearest(float *points, int *point_size,
                            int *width, int *height, float *scale_x, float *scale_y,
                            int* nearest_label,
                            uint64_t *pixels_index,
                            int *points_index) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *point_size) {
        return;
    }
    float point_x = points[3 * idx];
    float point_y = points[3 * idx + 1];
    int x = static_cast<int>(point_x * (*scale_x));
    int y = static_cast<int>(point_y * (*scale_y));
    // int pixel_index = static_cast<int>(y) * *width + static_cast<int>(x);
    uint64_t pixel_index = static_cast<uint64_t>(y * *width + x);
    pixel_index <<= 28;
    pixel_index |= *((uint32_t*)&(nearest_label[idx]));
    pixels_index[idx] = pixel_index;
    points_index[idx] = idx;
}

__global__ void init_ranges_bev(uint2* ranges, int *width, int *height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= *width * *height) {
        return;
    }
    ranges[idx].x = 0;
    ranges[idx].y = 0;
}

__global__ void init_ranges_nearest(uint2* ranges, int *width, int *height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= *width * *height) {
        return;
    }
    ranges[idx].x = 0;
    ranges[idx].y = 0;
}

__global__ void identify_ranges(int* pixel_index, uint2* ranges, int *point_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= *point_size)
		return;
    int pixel = pixel_index[idx];
	if (idx == 0) {
		ranges[pixel].x = 0;
    }
	else {
		int pre_pixel = pixel_index[idx - 1];
		if (pixel != pre_pixel) {
			ranges[pre_pixel].y = idx;
			ranges[pixel].x = idx;
		}
	}
	if (idx == *point_size - 1) {
		ranges[pixel].y = *point_size;
    }
}

__global__ void identify_ranges_nearest(uint64_t* pixel_index, uint2* ranges, int *point_size) {
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

__global__ void render_bev(unsigned char *bev_image, unsigned char *bev_label,
                       int *width, int *height,
                       float *rgbs, int *labels,
                       uint2 *ranges, int *points_index) {
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y;
    int pixel_index = pixel_x + pixel_y * *width;
    uint2 range = ranges[pixel_index];
    float3 rgb = {0.0, 0.0, 0.0};
    int label[6] = {0};
    uint2 label_max = {0, 0};
    for (int i = range.x; i < range.y; i++) {
        rgb.x += rgbs[3 * points_index[i]];
        rgb.y += rgbs[3 * points_index[i] + 1];
        rgb.z += rgbs[3 * points_index[i] + 2];
        label[labels[points_index[i]]]++;
        if (label[labels[points_index[i]]] > label_max.y) {
            label_max.x = labels[points_index[i]];
            label_max.y = label[labels[points_index[i]]]++;
        }
    }
    if (range.y > range.x) {
        bev_image[3 * pixel_index] = rgb.x / (range.y - range.x);
        bev_image[3 * pixel_index + 1] = rgb.y / (range.y - range.x);
        bev_image[3 * pixel_index + 2] = rgb.z / (range.y - range.x);
        bev_label[pixel_index] = label_max.x;
    }
    else {
        float3 rgb = {0.0, 0.0, 0.0};
        int count = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (pixel_y + i < 0 || pixel_y + i >= *height || 
                    pixel_x + j < 0 || pixel_x + j >= *width) {
                    continue;
                }
                uint2 range_k = ranges[pixel_x + j + (pixel_y + i) * *width];
                for (int k = range_k.x; k < range_k.y; k++) {
                    rgb.x += rgbs[3 * points_index[k]];
                    rgb.y += rgbs[3 * points_index[k] + 1];
                    rgb.z += rgbs[3 * points_index[k] + 2];
                    count++;
                }
            }
        }
        if (count > 0) {
            bev_image[3 * pixel_index] = rgb.x / count;
            bev_image[3 * pixel_index + 1] = rgb.y / count;
            bev_image[3 * pixel_index + 2] = rgb.z / count;
        }
    }
}

__global__ void render_bev_nearest(unsigned char *bev_image, unsigned char *bev_label_nearest,
                       int *width, int *height,
                       float *rgbs, int* labels_nearest,
                       uint2 *ranges, int *points_index) {
    int pixel_x = blockIdx.x;
    int pixel_y = blockIdx.y;
    int pixel_index = pixel_x + pixel_y * *width;
    uint2 range = ranges[pixel_index];
    float3 rgb = {0.0, 0.0, 0.0};
    if (range.y > range.x) {
        int mostFrequentNum = labels_nearest[points_index[range.x]];
        int maxCount = 1;
        int currentCount = 1;
        for (size_t i = range.x + 1; i < range.y; ++i) {
            if (labels_nearest[points_index[i]] == labels_nearest[points_index[i - 1]]) {
                currentCount++;
            } else {
                if (currentCount > maxCount) {
                    mostFrequentNum = labels_nearest[points_index[i - 1]];
                    maxCount = currentCount;
                }
                currentCount = 1;
            }
        }
        if (currentCount > maxCount) {
            mostFrequentNum = labels_nearest[points_index[range.y - 1]];
            maxCount = currentCount;
        }
        for (int i = range.x; i < range.y; i++) {
            rgb.x += rgbs[3 * points_index[i]];
            rgb.y += rgbs[3 * points_index[i] + 1];
            rgb.z += rgbs[3 * points_index[i] + 2];
        }
        bev_image[3 * pixel_index] = rgb.x / (range.y - range.x);
        bev_image[3 * pixel_index + 1] = rgb.y / (range.y - range.x);
        bev_image[3 * pixel_index + 2] = rgb.z / (range.y - range.x);
        bev_label_nearest[pixel_index] = mostFrequentNum;
    }
    else {
        float3 rgb = {0.0, 0.0, 0.0};
        int count = 0;
        int count_labels[67]={0};
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (pixel_y + i < 0 || pixel_y + i >= *height || 
                    pixel_x + j < 0 || pixel_x + j >= *width) {
                    continue;
                }
                uint2 range_k = ranges[pixel_x + j + (pixel_y + i) * *width];
                for (int k = range_k.x; k < range_k.y; k++) {
                    rgb.x += rgbs[3 * points_index[k]];
                    rgb.y += rgbs[3 * points_index[k] + 1];
                    rgb.z += rgbs[3 * points_index[k] + 2];
                    count++;
                    count_labels[labels_nearest[points_index[k]]]++;
                }
            }
        }
        if (count > 0) {
            bev_image[3 * pixel_index] = rgb.x / count;
            bev_image[3 * pixel_index + 1] = rgb.y / count;
            bev_image[3 * pixel_index + 2] = rgb.z / count;
            int best_label=0;
            int best_count=0;
            for(int i=0; i<67; ++i){
                if(count_labels[i]>best_count){
                    best_count=count_labels[i];
                    best_label=i;
                }
            }
            bev_label_nearest[pixel_index]=best_label;
        }
    }
}

DistributeProjector::DistributeProjector(float* h_points, float* h_rgbs,
                                         float* h_count, size_t num_points,
                                         const std::vector<int>& label_map,
                                         int* h_labels) {
    cudaMalloc((void**)&points_, 3 * num_points * sizeof(float));
    cudaMalloc((void**)&rgbs_, 3 * num_points * sizeof(float));
    cudaMalloc((void**)&rgb_counts_, num_points * sizeof(float));
    cudaMalloc((void**)&labels_, num_points * sizeof(int));
    cudaMalloc((void**)&nearest_label_, num_points * sizeof(int));
    cudaMalloc((void**)&nearest_z_, num_points * sizeof(float));
    cudaMalloc((void**)&points_size_, sizeof(int));
    cudaMalloc((void**)&label_map_, label_map.size() * sizeof(int));
    cudaMalloc((void**)&label0_counts_, num_points * sizeof(int));
    cudaMalloc((void**)&label1_counts_, num_points * sizeof(int));
    cudaMalloc((void**)&label2_counts_, num_points * sizeof(int));
    cudaMalloc((void**)&label3_counts_, num_points * sizeof(int));
    cudaMalloc((void**)&label4_counts_, num_points * sizeof(int));
    cudaMalloc((void**)&label5_counts_, num_points * sizeof(int));

    int* label_n_counts = new int[num_points];
    float* init_min_z = new float[num_points];
    for (int i = 0; i < num_points; i++) {
        label_n_counts[i] = 0;
        init_min_z[i] = 1000.0;
    }
    cudaMemcpy(points_, h_points, 3 * num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rgbs_, h_rgbs, 3 * num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rgb_counts_, h_count, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(labels_, h_labels, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label_map_, label_map.data(), label_map.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label0_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label1_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label2_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label3_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label4_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(label5_counts_, label_n_counts, num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(nearest_label_, 0, num_points * sizeof(int));
    cudaMemcpy(nearest_z_, init_min_z, num_points * sizeof(float), cudaMemcpyHostToDevice);
    int tmp = num_points;
    cudaMemcpy(points_size_, &tmp, sizeof(int), cudaMemcpyHostToDevice);
    size_ = num_points;
    delete[] label_n_counts;
    delete[] init_min_z;
}

void DistributeProjector::TransformPointToM(double* h_rotation, double* h_translation, float* h_points_m) {
    double* rotation, *translation;
    float* points_m;
    cudaMalloc((void**)&points_m, 3 * size_ * sizeof(float));
    cudaMalloc((void**)&rotation, 9 * sizeof(double));
    cudaMalloc((void**)&translation, 3 * sizeof(double));

    cudaMemcpy(rotation, h_rotation, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(translation, h_translation, 3 * sizeof(double), cudaMemcpyHostToDevice);
    transform_points_to_m<<< (size_ + 255) / 256, 256 >>>(points_m, points_, rotation, translation, points_size_);
    float* tmp = points_;
    points_ = points_m;
    cudaMemcpy(h_points_m, points_m, 3 * size_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tmp);
    cudaFree(rotation);
    cudaFree(translation);
}

void DistributeProjector::Project(double* h_rotation, double* h_translation,
                                  unsigned char* h_image, unsigned char* h_image_mask, unsigned char* h_image_seg,
                                  double* h_K,
                                  int h_height, int h_width) {
    double *rotation, *translation, *K;
    unsigned char *image, *image_mask, *image_seg;
    int *height, *width;
    // std::cout << "============ begin memery malloc and copy =============" << std::endl;
    cudaMalloc((void**)&rotation, 9 * sizeof(double));
    cudaMalloc((void**)&translation, 3 * sizeof(double));
    cudaMalloc((void**)&K, 9 * sizeof(double));
    cudaMalloc((void**)&image, 3 * (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&image_mask, (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&image_seg, (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&height, sizeof(int));
    cudaMalloc((void**)&width, sizeof(int));

    cudaMemcpy(rotation, h_rotation, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(translation, h_translation, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(image, h_image, 3 * (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image_mask, h_image_mask, (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image_seg, h_image_seg, (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(K, h_K, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(height, &h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(width, &h_width, sizeof(int), cudaMemcpyHostToDevice);
    int** h_label_counts = new int*[6];
    int** label_counts;
    cudaMalloc((void**)&label_counts, 6 * sizeof(int*));
    h_label_counts[0] = label0_counts_;
    h_label_counts[1] = label1_counts_;
    h_label_counts[2] = label2_counts_;
    h_label_counts[3] = label3_counts_;
    h_label_counts[4] = label4_counts_;
    h_label_counts[5] = label5_counts_;
    cudaMemcpy(label_counts, h_label_counts, 6 * sizeof(int*), cudaMemcpyHostToDevice);
    // std::cout << "============ end memery malloc and copy =============" << std::endl;
    project_cuda<<< (size_ + 255) / 256, 256 >>> (rotation, translation,
                                                  image, image_mask, image_seg,
                                                  height, width, K,
                                                  points_, rgbs_, rgb_counts_,
                                                  label_map_,
                                                  label_counts,
                                                  points_size_);
    cudaDeviceSynchronize();
    cudaFree(rotation);
    cudaFree(translation);
    cudaFree(image);
    cudaFree(image_mask);
    cudaFree(image_seg);
    cudaFree(height);
    cudaFree(width);
    cudaFree(K);
    delete[] h_label_counts;
}

void DistributeProjector::ResetRGB() {
    cudaMemset(rgbs_, 0.0, 3 * size_ * sizeof(float));
    cudaMemset(rgb_counts_, 0.0, size_ * sizeof(float));
}

void DistributeProjector::ProjectNearest(double* h_rotation, double* h_translation,
                                  unsigned char* h_image, unsigned char* h_image_mask, unsigned char* h_image_seg,
                                  double* h_K,
                                  int h_height, int h_width) {
    double *rotation, *translation, *K;
    unsigned char *image, *image_mask, *image_seg;
    int *height, *width;
    // std::cout << "============ begin memery malloc and copy =============" << std::endl;
    cudaMalloc((void**)&rotation, 9 * sizeof(double));
    cudaMalloc((void**)&translation, 3 * sizeof(double));
    cudaMalloc((void**)&K, 9 * sizeof(double));
    cudaMalloc((void**)&image, 3 * (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&image_mask, (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&image_seg, (h_height * h_width) * sizeof(unsigned char));
    cudaMalloc((void**)&height, sizeof(int));
    cudaMalloc((void**)&width, sizeof(int));

    cudaMemcpy(rotation, h_rotation, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(translation, h_translation, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(image, h_image, 3 * (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image_mask, h_image_mask, (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(image_seg, h_image_seg, (h_height * h_width) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(K, h_K, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(height, &h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(width, &h_width, sizeof(int), cudaMemcpyHostToDevice);
    // std::cout << "============ end memery malloc and copy =============" << std::endl;
    project_cuda_nearest<<< (size_ + 255) / 256, 256 >>> (rotation, translation,
                                                  image, image_mask, image_seg,
                                                  height, width, K,
                                                  points_, rgbs_, rgb_counts_,
                                                  nearest_label_,
                                                  nearest_z_,
                                                  points_size_);
    cudaDeviceSynchronize();
    cudaFree(rotation);
    cudaFree(translation);
    cudaFree(image);
    cudaFree(image_mask);
    cudaFree(image_seg);
    cudaFree(height);
    cudaFree(width);
    cudaFree(K);
}

void DistributeProjector::Mix(float *h_rgbs, int *labels) {
   int** h_label_counts = new int*[6];
   int** label_counts;
   cudaMalloc((void**)&label_counts, 6 * sizeof(int*));
   h_label_counts[0] = label0_counts_;
   h_label_counts[1] = label1_counts_;
   h_label_counts[2] = label2_counts_;
   h_label_counts[3] = label3_counts_;
   h_label_counts[4] = label4_counts_;
   h_label_counts[5] = label5_counts_;
   cudaMemcpy(label_counts, h_label_counts, 6 * sizeof(int*), cudaMemcpyHostToDevice);
   mix_color_and_label_cuda<<< (size_ + 255) / 256, 256 >>>(points_, rgbs_, labels_, rgb_counts_, label_counts, points_size_);
   cudaDeviceSynchronize();
   cudaMemcpy(h_rgbs, rgbs_, 3 * size_ * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(labels, labels_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
   // cudaFree(rgb_counts_);
   cudaFree(label0_counts_);
   cudaFree(label1_counts_);
   cudaFree(label2_counts_);
   cudaFree(label3_counts_);
   cudaFree(label4_counts_);
   cudaFree(label5_counts_);
   delete[] h_label_counts;
}

void DistributeProjector::MixNearest(float *h_rgbs, int *labels) {
   mix_color_cuda_nearest<<< (size_ + 255) / 256, 256 >>>(points_, rgbs_, rgb_counts_, points_size_);
   cudaDeviceSynchronize();
   cudaMemcpy(h_rgbs, rgbs_, 3 * size_ * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(labels, nearest_label_, size_ * sizeof(int), cudaMemcpyDeviceToHost);
}

void DistributeProjector::BevImage(int *h_width, int *h_height, float *h_scale_x, float *h_scale_y,
                                   unsigned char *h_bev_image, unsigned char *h_bev_label) {
    int *width, *height;
    float *scale_x, *scale_y;
    unsigned char *bev_image, *bev_label, *bev_label_nearest;
    int *pixels_index, *points_index;
    int *pixels_index_sorted, *points_index_sorted;
    uint2* ranges;
    cudaMalloc((void**)&width, sizeof(int));
    cudaMalloc((void**)&height, sizeof(int));
    cudaMalloc((void**)&scale_x, sizeof(float));
    cudaMalloc((void**)&scale_y, sizeof(float));
    cudaMalloc((void**)&bev_image, 3 * *h_width * *h_height * sizeof(unsigned char));
    cudaMalloc((void**)&bev_label, *h_width * *h_height * sizeof(unsigned char));
    cudaMalloc((void**)&bev_label_nearest, *h_width * *h_height * sizeof(unsigned char));

    cudaMalloc((void**)&pixels_index, (size_) * sizeof(int));
    cudaMalloc((void**)&points_index, (size_) * sizeof(int));
    cudaMalloc((void**)&pixels_index_sorted, (size_) * sizeof(int));
    cudaMalloc((void**)&points_index_sorted, (size_) * sizeof(int));
    cudaMalloc((void**)&ranges, (*h_width) * (*h_height) * sizeof(uint2));

    cudaMemcpy(width, h_width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(height, h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_x, h_scale_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_y, h_scale_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(bev_image, 0, 3 * *h_width * *h_height * sizeof(unsigned char));
    cudaMemset(bev_label, 0, *h_width * *h_height * sizeof(unsigned char));
    cudaMemset(bev_label_nearest, 0, *h_width * *h_height * sizeof(unsigned char));

    point_pixel_index<<< (size_ + 255) / 256, 256 >>>(points_, points_size_, width, height, scale_x, scale_y, pixels_index, points_index);
    cudaDeviceSynchronize();
    void *sort_storage_tmp = nullptr;
    size_t sort_storage_size = 0;
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixels_index, pixels_index_sorted, points_index, points_index_sorted, size_);
    cudaMalloc(&sort_storage_tmp, sort_storage_size);
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixels_index, pixels_index_sorted, points_index, points_index_sorted, size_);
    cudaDeviceSynchronize();
    // test<<< 1, 1 >>>(pixel_index_sorted, points_index_sorted);
    init_ranges_bev<<< (*h_width * *h_height + 255) / 256 , 256>>>(ranges, width, height);
    identify_ranges<<< (size_ + 255) / 256, 256 >>> (pixels_index_sorted, ranges, points_size_);
    dim3 grid_dim(*h_width, *h_height);
    render_bev<<< grid_dim, 1 >>>(bev_image, bev_label, width, height, rgbs_, labels_, ranges, points_index_sorted);
    cudaDeviceSynchronize();
    cudaMemcpy(h_bev_image, bev_image, 3 * *h_width * *h_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bev_label, bev_label, *h_width * *h_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(sort_storage_tmp);
    cudaFree(pixels_index);
    cudaFree(points_index);
    cudaFree(labels_);
    cudaFree(label_map_);
}
void DistributeProjector::BevImageNearest(int *h_width, int *h_height, float *h_scale_x, float *h_scale_y,
                                   unsigned char *h_bev_image, unsigned char *h_bev_label_nearest) {
    int *width, *height;
    float *scale_x, *scale_y;
    unsigned char *bev_image, *bev_label;
    uint64_t *pixels_index, *pixels_index_sorted;
    int *points_index, *points_index_sorted;
    uint2* ranges;
    cudaMalloc((void**)&width, sizeof(int));
    cudaMalloc((void**)&height, sizeof(int));
    cudaMalloc((void**)&scale_x, sizeof(float));
    cudaMalloc((void**)&scale_y, sizeof(float));
    cudaMalloc((void**)&bev_image, 3 * *h_width * *h_height * sizeof(unsigned char));
    cudaMalloc((void**)&bev_label, *h_width * *h_height * sizeof(unsigned char));
    // cudaMalloc((void**)&bev_label_nearest, *h_width * *h_height * sizeof(unsigned char));

    cudaMalloc((void**)&pixels_index, (size_) * sizeof(uint64_t));
    cudaMalloc((void**)&points_index, (size_) * sizeof(int));
    cudaMalloc((void**)&pixels_index_sorted, (size_) * sizeof(uint64_t));
    cudaMalloc((void**)&points_index_sorted, (size_) * sizeof(int));
    cudaMalloc((void**)&ranges, (*h_width) * (*h_height) * sizeof(uint2));

    cudaMemcpy(width, h_width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(height, h_height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_x, h_scale_x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_y, h_scale_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(bev_image, 0, 3 * *h_width * *h_height * sizeof(unsigned char));
    cudaMemset(bev_label, 0, *h_width * *h_height * sizeof(unsigned char));

    point_pixel_index_nearest<<< (size_ + 255) / 256, 256 >>>(points_, points_size_, width, height, scale_x, scale_y, nearest_label_, pixels_index, points_index);
    cudaDeviceSynchronize();
    void *sort_storage_tmp = nullptr;
    size_t sort_storage_size = 0;
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixels_index, pixels_index_sorted, points_index, points_index_sorted, size_);
    cudaMalloc(&sort_storage_tmp, sort_storage_size);
    cub::DeviceRadixSort::SortPairs(sort_storage_tmp, sort_storage_size, pixels_index, pixels_index_sorted, points_index, points_index_sorted, size_);
    cudaDeviceSynchronize();
    init_ranges_nearest<<< (*h_width * *h_height + 255) / 256 , 256>>>(ranges, width, height);
    identify_ranges_nearest<<< (size_ + 255) / 256, 256 >>> (pixels_index_sorted, ranges, points_size_);
    dim3 grid_dim(*h_width, *h_height);
    render_bev_nearest<<< grid_dim, 1 >>>(bev_image, bev_label, width, height, rgbs_, nearest_label_, ranges, points_index_sorted);
    cudaDeviceSynchronize();
    cudaMemcpy(h_bev_image, bev_image, 3 * *h_width * *h_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bev_label_nearest, bev_label, *h_width * *h_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(sort_storage_tmp);
    cudaFree(pixels_index);
    cudaFree(points_index);

}

DistributeProjector::~DistributeProjector() {
    cudaFree(points_);
    cudaFree(rgbs_);
    cudaFree(rgb_counts_);
    cudaFree(nearest_label_);
    cudaFree(nearest_z_);
    cudaFree(points_size_);
}
