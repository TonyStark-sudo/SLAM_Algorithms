#include "pose.h"


Pose::Pose() : p(Eigen::Vector3d::Zero()), q(Eigen::Quaterniond::Identity()) {}

Pose::Pose(const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat)
    : p(pos), q(quat.normalized()) {}

Pose Pose::operator*(const Pose &other) const {
    Pose temp;
    temp.p = p + q * other.p;
    temp.q = q * other.q;
    return temp;
}

Pose Pose::Inverse() const {
    Pose temp;
    temp.q = q.inverse();
    temp.p = q.inverse() * (-p);
    return temp;
}

Eigen::Vector3d Pose::operator*(const Eigen::Vector3d &v) const { return q * v + p; }

PoseStamped::PoseStamped(double _stamp, const std::string&stamp_str, const Pose &_pose)
    : timestamp(_stamp), timestamp_str(stamp_str), pose(_pose) {}
PoseStamped::PoseStamped(double _stamp, const std::string&stamp_str, const Eigen::Vector3d &p,
                         const Eigen::Quaterniond &q)
    : timestamp(_stamp), timestamp_str(stamp_str), pose(p, q) {}
