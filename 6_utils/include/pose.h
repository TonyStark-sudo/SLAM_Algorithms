#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>

struct Pose {
    Pose();
    Pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q);
    Eigen::Vector3d p;
    Eigen::Quaterniond q;

    Pose operator* (const Pose& other) const;
    Eigen ::Vector3d operator* (const Eigen::Vector3d& vec) const;
    Pose inverse() const;
};

struct PoseStamped {
    double timestamp;
    std::string timestamp_str;
    Pose pose;
    PoseStamped() = default;
    PoseStamped(double stamp, const std::string& stamp_str, const Pose& pose);
    PoseStamped(double stamp, const std::string& stamp_str, const Eigen::Vector3d& p, const Eigen::Quaterniond& q);
};