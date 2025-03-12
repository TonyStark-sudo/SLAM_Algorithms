#pragma once

#include <string>
#include <filesystem>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>


class Projector {
 public:
    Projector(const std::string& config);
    void ProjectColorful();
 private:
    template <typename... Paths> 
    std::filesystem::path JoinPaths(const std::filesystem::path& base, const Paths&... parts) {
        std::filesystem::path result = base; 
        (result /= ... /= parts); 
        return result; 
    }                                                                                          
    Eigen::Vector2d BrownConradyPhiholeProjection(const Eigen::Vector3d& X,
                                                  const std::vector<double>& D,
                                                  const Eigen::Matrix3d& K);
    double* mat3x3_to_pointer(const Eigen::Matrix3d& K);
    double* vec3x1_to_pointer(const Eigen::Vector3d& K);
    std::vector<cv::Vec3b> color_map_;
    std::vector<cv::Vec3b> color_map_all_label_;
    std::vector<int> label_map_;
    nlohmann::json config_;
};
