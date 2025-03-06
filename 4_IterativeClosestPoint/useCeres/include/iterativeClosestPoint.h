#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

class CreateICPProblem {
public:
    CreateICPProblem(const cv::Mat &_img_1, const cv::Mat &_img_2, const cv::Mat &_dpt_1, const cv::Mat &_dpt_2) : 
                    img_1(_img_1), img_2(_img_2), dpt_1(_dpt_1), dpt_2(_dpt_2) {}
    
    ~CreateICPProblem();

    void find_feature_matches(std::vector<cv::KeyPoint> &kps_1, 
                              std::vector<cv::KeyPoint> &kps_2, 
                              std::vector<cv::DMatch> &matches);

    std::vector<cv::Point3d> pixel2cam(const std::vector<cv::Point2d> &p, const cv::Mat &K);

private:
    const cv::Mat img_1;
    const cv::Mat img_2;
    const cv::Mat dpt_1;
    const cv::Mat dpt_2;
    
};

class BundleAdjustment {
public:
    
};

