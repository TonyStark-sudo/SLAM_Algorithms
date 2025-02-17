#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>

// 相机内参，假设已知（例如焦距和主点位置）
const double fx = 800.0; // x轴焦距
const double fy = 800.0; // y轴焦距
const double cx = 320.0; // 图像中心x
const double cy = 240.0; // 图像中心y

// 3D点到2D图像平面的重投影误差
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, 
                      const Eigen::Vector3d& point_3d, 
                      const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : observed_x_(observed_x), observed_y_(observed_y), 
          point_3d_(point_3d), R_(R), t_(t) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const {
        // 从相机参数中获取旋转矩阵和平移向量
        Eigen::Matrix<T, 3, 3> R;
        Eigen::Matrix<T, 3, 1> t;

        // 从相机参数获取旋转矩阵和平移向量
        for (int i = 0; i < 9; ++i) {
            R(i / 3, i % 3) = T(camera[i]);
        }
        for (int i = 0; i < 3; ++i) {
            t(i) = T(camera[9 + i]);
        }

        // 3D点转换到相机坐标系
        Eigen::Matrix<T, 3, 1> point_3d;
        for (int i = 0; i < 3; ++i) {
            point_3d(i) = T(point_3d_(i));
        }
        
        // 将3D点从世界坐标系转换到相机坐标系
        Eigen::Matrix<T, 3, 1> point_camera = R * point_3d + t;

        // 重投影
        T predicted_x = T(fx) * point_camera(0) / point_camera(2) + T(cx);
        T predicted_y = T(fy) * point_camera(1) / point_camera(2) + T(cy);

        // 计算重投影误差
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y, 
                                       const Eigen::Vector3d& point_3d, 
                                       const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12>(
                    new ReprojectionError(observed_x, observed_y, point_3d, R, t)));
    }

private:
    double observed_x_;
    double observed_y_;
    Eigen::Vector3d point_3d_;
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;
};

// Bundle Adjustment函数
void bundleAdjustment(std::vector<Eigen::Vector3d>& points_3d, 
                      std::vector<Eigen::Vector2d>& observations, 
                      Eigen::Matrix3d& R, Eigen::Vector3d& t) {

    // 初始化相机参数（旋转矩阵和平移向量）
    std::vector<double> camera(12);  // 9个旋转矩阵元素 + 3个平移向量元素
    for (int i = 0; i < 9; ++i) {
        camera[i] = R(i / 3, i % 3);
    }
    for (int i = 0; i < 3; ++i) {
        camera[9 + i] = t(i);
    }

    // 创建优化问题
    ceres::Problem problem;

    for (size_t i = 0; i < points_3d.size(); ++i) {
        ceres::CostFunction* cost_function = ReprojectionError::Create(
            observations[i][0], observations[i][1], points_3d[i], R, t);
        problem.AddResidualBlock(cost_function, nullptr, camera.data());
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 更新优化后的相机位姿
    for (int i = 0; i < 9; ++i) {
        R(i / 3, i % 3) = camera[i];
    }
    for (int i = 0; i < 3; ++i) {
        t(i) = camera[9 + i];
    }

    // 更新优化后的3D点
    for (size_t i = 0; i < points_3d.size(); ++i) {
        points_3d[i] = Eigen::Vector3d(camera[3 * i], camera[3 * i + 1], camera[3 * i + 2]);
    }
}

int main() {
    // 示例输入数据
    std::vector<Eigen::Vector3d> points_3d = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    std::vector<Eigen::Vector2d> observations = {
        {100.0, 100.0},
        {200.0, 200.0},
        {300.0, 300.0}
    };
    Eigen::Matrix3d R_init = Eigen::Matrix3d::Identity();  // 初始旋转矩阵
    Eigen::Vector3d t_init(0.0, 0.0, 0.0);  // 初始平移向量

    // 执行Bundle Adjustment
    bundleAdjustment(points_3d, observations, R_init, t_init);

    // 输出优化结果
    std::cout << "Optimized 3D points:" << std::endl;
    for (const auto& point : points_3d) {
        std::cout << point.transpose() << std::endl;
    }
    std::cout << "Optimized camera pose:" << std::endl;
    std::cout << "Rotation matrix:\n" << R_init << std::endl;
    std::cout << "Translation vector: " << t_init.transpose() << std::endl;

    return 0;
}
