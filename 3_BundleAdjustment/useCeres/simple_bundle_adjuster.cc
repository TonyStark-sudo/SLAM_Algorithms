
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
 public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }
  // num_observation: number of observations
  // 调用成员函数num_observations，返回num_observations_
  int num_observations()       const { return num_observations_;               }

  // 调用成员函数observations返回observations_的首地址
  const double* observations() const { return observations_;                   }


  // 下面的指针用来存储优化后的参数值
  // 调用成员函数mutable_cameras返回parameters_的首地址
  double* mutable_cameras()          { return parameters_;                     }

  // 调用mutable_points返回存储3D点坐标的首地址，从地址为 &parameters_[9 * num_cameras_] 开始
  double* mutable_points()           { return parameters_  + 9 * num_cameras_; }

  // 返回第i个观测对应的相机参数,用于保存优化后的相机参数
  // camera_index_[i] 的size是16, 所以有传入相同待优化相机参数地址的残差块
  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9;
  }

  // 返回第i个观测对应的3D点坐标，用于保存优化后的3D点坐标
  // points_index_[i] 的size是22106, 所以有传入相同待优化3D点坐标地址的残差块
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  // 导入数据，对一些变量进行赋值
  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

    // num_cameras_ = 16
    FscanfOrDie(fptr, "%d", &num_cameras_);
    // num_points_ = 22106 
    FscanfOrDie(fptr, "%d", &num_points_);
    // num_observations_ = 83718
    FscanfOrDie(fptr, "%d", &num_observations_);

    // size = 83718
    point_index_ = new int[num_observations_];
    // size = 83718
    camera_index_ = new int[num_observations_];
    // size = 83718 * 2
    observations_ = new double[2 * num_observations_];

    // 待优化的参数个数 9 * 16 + 3 * 22106 = 66462
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // 读取观测数据，一共 83718 行，对应 83718 个uv坐标
    for (int i = 0; i < num_observations_; ++i) {
      // 每个uv坐标对应的相机索引
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      // 每个uv坐标对应的3D点索引
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        // 每个uv坐标的值
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }
    // 读取优化参数初值，一共 66462 行，对应 66462 个参数
    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

  void SaveOptimizedParameters(const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
      std::cerr << "Error: unable to open file " << filename << "\n";
      return;
    }

    out_file << "optimized Camera Parameters: \n";
    for (int i = 0; i < num_cameras_; i++) {
      out_file << "Camera_id_" << i << ": \n"; 
      out_file << "angular_axis: ";
      for (int j = 0; j < 3; j++) {
        out_file << parameters_[i * 9 + j] << " ";
      }
      out_file << "trans x y z: ";
      for (int j = 3; j < 6; j++) {
        out_file << parameters_[i * 9 + j] << " ";
      }
      out_file << "focal k1 k2: "; 
      for (int j = 6; j < 9; j++) {
        out_file << parameters_[i * 9 + j] << " ";
      }
      out_file << "\n";

    }

    out_file << "\n3D_points coordinates: \n";
    for (int i = 0; i < num_points_; i++) {
      out_file << "Point_" << i << ": (";
      for (int j = 0; j < 3; j++) {
        out_file << parameters_[num_cameras_ * 9 + 3 * i + j] << " ";
      }
      out_file << ")\n";
    }
  }

 private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-num_observationsaxis rotation.
    T p[3];

    // 将所有的3D点旋转到相机坐标系下保存到p中
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    // 将所有的3D点平移到相机坐标系下保存到p中

    // 这时的p是相机坐标系下的3D点
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.

    // 对归一化坐标系下的点进行径向畸变校正
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    // 得到归一化坐标系下的预测点(带有畸变)
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }
  // 2 * 83718
  const double* observations = bal_problem.observations();

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;

  // bal_problem.num_observations();是观测的个数 83718，每个观测对应一个残差 residual 2 * 1
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  // 它利用了 BA 问题的稀疏性和结构，将大规模稀疏问题转化为较小的稠密 Schur 补系统进行求解，
  // 效率高，适合相机数量较少、点数量较多的场景。
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  std::string output_file = "BA_optimized_parameters.txt";
  bal_problem.SaveOptimizedParameters(output_file);
  return 0;
}
