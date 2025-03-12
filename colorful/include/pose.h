#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

struct Pose {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    Pose();
    Pose(const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat);
    Pose operator*(const Pose &other) const;
    Pose Inverse() const;
    Eigen::Vector3d operator*(const Eigen::Vector3d &v) const;
};

struct PoseStamped {
    double timestamp;
    std::string timestamp_str;
    Pose pose;

    PoseStamped() = default;
    PoseStamped(double stamp, const std::string&stamp_str, const Pose &pose);
    PoseStamped(double stamp, const std::string&stamp_str, const Eigen::Vector3d &p, const Eigen::Quaterniond &q);
};
