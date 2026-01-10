#include <vector>
// #include "absl/log/initialize.h"
#include <ceres/ceres.h>
#include <iostream>

class MyCostFunction : public ceres::SizedCostFunction<1, 1> {
public:
    virtual ~MyCostFunction() {}
    virtual bool Evaluate(double const* const* parameters, 
                          double* residuals, 
                          double** jacobians) const {
        const double x = parameters[0][0];
        residuals[0] = 10 - x;

        // Compute the Jacobian if asked for.
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -1;
        }
        return true;
    }
};

int main(int argc, char** argv) {
//   absl::InitializeLog();
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;
  // Build the problem.
  ceres::Problem problem;
  // Set up the only cost function (also known as residual).
  ceres::CostFunction* cost_function = new MyCostFunction;
  problem.AddResidualBlock(cost_function, nullptr, &x);
  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}