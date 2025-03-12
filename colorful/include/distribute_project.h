#pragma once

#include <cuda_runtime.h>
#include <vector>

class DistributeProjector {
  public:
    DistributeProjector(float* h_points, float* h_rgbs, float* h_count, size_t num_points,
                        const std::vector<int>& label_map, int* labels);
    ~DistributeProjector();

    void TransformPointToM(double* rotation, double* translation, float* h_points_m);
    void Project(double* rotation, double* translation,
                 unsigned char * image, unsigned char* image_mask, unsigned char* image_seg,
                 double* K, int height, int width);
    void ProjectNearest(double* rotation, double* translation,
                 unsigned char * image, unsigned char* image_mask, unsigned char* image_seg,
                 double* K, int height, int width);
    void ResetRGB();
    void Mix(float *h_rgbs, int *labels);
    void MixNearest(float *h_rgbs, int *labels);
    void BevImage(int *width, int *height, float *scale_x, float *scale_y,
                  unsigned char *bev_image, unsigned char *bev_label);
    void BevImageNearest(int *width, int *height, float *scale_x, float *scale_y,
                  unsigned char *bev_image, unsigned char *bev_label_nearest);

    float* points_;
    float* rgbs_;
    float* rgb_counts_;

    int* labels_;
    int* label_map_;
    int *label0_counts_;
    int *label1_counts_;
    int *label2_counts_;
    int *label3_counts_;
    int *label4_counts_;
    int *label5_counts_;

    // Note(Xiaoming): Front-viewing camera, the latest is the nearest
    int *nearest_label_;
    float *nearest_z_;

    int* points_size_;
    int size_;
};
