#include "pose.h"

Pose::Pose() : p(Eigen::Vector3d::Zero()), q(Eigen::Quaterniond::Identity()) {}
Pose::Pose(const Eigen::Vector3d& p_, const Eigen::Quaterniond& q_) : p(p_), q(q_.normalized()) {}

Pose Pose::operator* (const Pose& other) const {
    Pose res;
    res.q = q * other.q;
    res.p = q * other.p + p;
    return res;
}

Eigen::Vector3d Pose::operator* (const Eigen::Vector3d& vec) const {
    return q * vec + p;
}

Pose Pose::inverse() const {
    Pose res;
    res.q = q.inverse();
    res.p = q.inverse() * (- p);
    return res;
}

PoseStamped::PoseStamped(double timestamp_, const std::string& timestamp_str_, const Pose& pose_) :
    timestamp(timestamp_), timestamp_str(timestamp_str_), pose(pose_) {}

PoseStamped::PoseStamped(double timestamp_, const std::string& timestamp_str_, const Eigen::Vector3d& p_, const Eigen::Quaterniond& q_) :
    timestamp(timestamp_), timestamp_str(timestamp_str_), pose(p_, q_) {}