/*
1. 传感器数据处理：IMU积分
题目描述：
给定一组IMU（惯性测量单元）数据，包含时间戳、角速度和线性加速度。编写一个函数，计算在给定时间段内的姿态（旋转）和位置变化。

输入：

imu_data: 一个列表，每个元素是一个包含 timestamp, angular_velocity, linear_acceleration 的字典。

initial_pose: 初始姿态（四元数或旋转矩阵）和位置（3D向量）。

输出：

final_pose: 积分后的最终姿态和位置。

要求：

使用欧拉积分或中值积分方法。

考虑重力对加速度的影响。
*/

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct IMUData {
    double timestamps;
    Eigen::Vector3d ang_velo;
    Eigen::Vector3d linear_acc;
};

void intergrateIMU(const std::vector<IMUData>& imu_data, 
                    Eigen::Quaterniond& q, Eigen::Vector3d& p, 
                    Eigen::Vector3d& v) {

    Eigen::Vector3d gravity(0, 0, -9.81);
    for (size_t i = 1; i < imu_data.size(); i++) {
        double dt = imu_data[i].timestamps - imu_data[i - 1].timestamps;
        if (dt <= 0) 
            continue;
        
        Eigen::Vector3d w = imu_data[i - 1].ang_velo;

        // 欧拉积分更新四元数
        Eigen::Quaterniond dq;
        Eigen::Vector3d w_half = 0.5 * w * dt;
        dq.w() = 1;
        dq.vec() = w_half;
        dq.normalize();
        q = q * dq;
        q.normalize();

        // 使用旋转矩阵， 其中w.normalized()是单位向量可以代表旋转轴，w.norm() * dt代表旋转角度
        // Eigen::Matrix3d R = Eigen::AngleAxisd(w.norm() * dt, w.normalized()).toRotationMatrix();
        // Eigen::Quaterniond dq(R);
        // q = q * dq;
        // q.normalize();

        // // 使用罗德里格斯公式
        // Eigen::AngleAxisd angle_axis(w.norm() * dt, w.normalized());
        // Eigen::Quaterniond dq(angle_axis);
        // q = q * dq;
        // q.normalize();

        // // 使用四元数微分方程的数值积分
        // Eigen::Quaterniond dq(1, 0.5 * w.x() * dt, 0.5 * w.y() * dt, 0.5 * w.z() * dt);
        // q = q * dq;
        // q.normalize();

        Eigen::Vector3d acc = imu_data[i - 1].linear_acc - gravity;

        // 为什么四元数和三维向量可以直接 * 操作？
        // 因为Eigen库重载了四元数和向量之间的乘法运算符，当写q * acc时，Eigen会自动将
        // 向量acc转换为四元数形式，然后执行四元数与向量的旋转计算，最后返回旋转后的向量

        Eigen::Vector3d acc_world = q * acc;

        v += acc_world * dt;
        p += v * dt + 0.5 * acc_world * dt * dt;   
    }
}

int main(int argc, char *argv[])
{

    Eigen::Quaterniond q(1, 0, 0, 0);
    Eigen::Vector3d position(0, 0, 0);
    Eigen::Vector3d velocity(0, 0, 0);

    std::vector<IMUData> imu_data = {
        {0.0, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)}, 
        {0.1, Eigen::Vector3d(0, 0, 0.1), Eigen::Vector3d(0.1, 0, 0)},
        {0.2, Eigen::Vector3d(0, 0, 0.2), Eigen::Vector3d(0.2, 0, 0)},
        {0.3, Eigen::Vector3d(0, 0, 0.3), Eigen::Vector3d(0.3, 0, 0)}  
    };
    intergrateIMU(imu_data, q, position, velocity);
    std::cout << "Final Position: " << position.transpose() << std::endl;
    std::cout << "Final Velocity: " << velocity.transpose() << std::endl;
    std::cout << "Final Quaternion: " << q.coeffs().transpose() << std::endl;
    /* code */
    return 0;
}

