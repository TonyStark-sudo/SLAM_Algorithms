#pragma once

#include <cuda_runtime.h>

class ImageProcess {
 public:
    void DepthImage(float* bev_points, int* point_size,
                    int* width, int* height, float* scale_x, float* scale_y,
                    float* bev_height, bool* bev_mask, int* uvs, int uv_length);
};
