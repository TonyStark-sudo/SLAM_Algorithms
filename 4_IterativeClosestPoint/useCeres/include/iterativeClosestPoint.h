#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class CreateICPProblem {
public:
    CreateICPProblem(const cv::Mat &_img_1, const cv::Mat &_img_2, const cv::Mat &_dpt_1, const cv::Mat &_dpt_2) : 
                    img_1(_img_1), img_2(_img_2), dpt_1(_dpt_1), dpt_2(_dpt_2) {}
    
    // ~CreateICPProblem();

    void find_feature_matches(std::vector<cv::KeyPoint> &kps_1, 
                              std::vector<cv::KeyPoint> &kps_2, 
                              std::vector<cv::DMatch> &matches);

    void get_3D_points(std::vector<cv::Point3f>& points_1, std::vector<cv::Point3f>& points_2, 
                       std::vector<cv::KeyPoint> &kps_1, std::vector<cv::KeyPoint> &kps_2,
                       std::vector<cv::DMatch> &matches);

private:
    const cv::Mat img_1;
    const cv::Mat img_2;
    const cv::Mat dpt_1;
    const cv::Mat dpt_2;

};

class BundleAdjustment {
public:

    BundleAdjustment(const std::vector<cv::Point3f>& points_1_, const std::vector<cv::Point3f>& points_2_)
        : points_1(points_1_), points_2(points_2_) {}

    ~BundleAdjustment() {
        delete[] t_init;
        delete[] q_init;
    }

    void bundleAdjustment();
    void getInitiValue(); 

    Eigen::Vector3d getTrans() {
        std::cout << "Optimized trans: " 
        << "(" << t_init[0] << ", " << t_init[1] << ", " << t_init[2] << ")\n";
        return Eigen::Vector3d(t_init[0], t_init[1], t_init[2]);
    }

    Eigen::Quaterniond getRota() { 
        double angle = sqrt(q_init[0] * q_init[0] + q_init[1] * q_init[1] + q_init[2] * q_init[2]);
        Eigen::Vector3d rota_vec(q_init[0], q_init[1], q_init[2]);
        rota_vec.normalize();
        Eigen::AngleAxisd angleaxis(angle, rota_vec);
        return Eigen::Quaterniond(angleaxis);
    }

private:
    double* t_init;
    double* q_init;
    const std::vector<cv::Point3f>& points_1;
    const std::vector<cv::Point3f>& points_2;

};