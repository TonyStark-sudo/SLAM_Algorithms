#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;

class KalmanFilter {
public:
    KalmanFilter(double process_noise, double measurement_noise, double dt, const Eigen::VectorXd& initial_state) 
    : process_noise_(process_noise), measurement_noise_(measurement_noise), dt_(dt), state_(initial_state) {
        
        P_ = Eigen::MatrixXd::Identity(2, 2);

        F_ = Eigen::MatrixXd(2, 2);
        F_ << 1.0, dt_,
              0.0, 1.0;
        
        H_ = Eigen::MatrixXd(1, 2);
        H_ << 1.0, 0.0;

        Q_ = Eigen::MatrixXd(2, 2);
        Q_ << dt_ * dt_ / 2.0, dt_,
              dt_, 1.0;

        R_ = Eigen::MatrixXd(1, 1);
        R_ << measurement_noise_;
    }

    vector<double> filter(const vector<double>& measurements) {
        vector<double> estimated_states;

        for (double z : measurements) {

            predict();

            update(z);
            estimated_states.push_back(state_(0));
        }

        return estimated_states;
    }

private:
    void predict() {
        state_ = F_ * state_;

        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    void update(double z) {
        Eigen::VectorXd y(1);
        y << z - (H_ * state_)(0, 0);

        Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
        Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

        state_ = state_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(2, 2) - K * H_) * P_;
    }

    double process_noise_;
    double measurement_noise_;
    double dt_;
    
    Eigen::VectorXd state_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd H_;

    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
};

int main(int argc, char *argv[])
{
    vector<double> measurements = {1.0, 2.0, 3.0, 4.0, 5.0};
    Eigen::VectorXd initial_state(2);
    initial_state << 0.0, 1.0;  
    double process_noise = 0.1;
    double measurement_noise = 0.1;
    double dt = 1.0;  // 时间步长
    KalmanFilter filter(process_noise, measurement_noise, dt, initial_state);
    vector<double> estimated_states = filter.filter(measurements);
    
    cout << "Estimated positions: ";
    for (double pos : estimated_states) {
        cout << pos << " ";
    }
    cout << endl;  
    return 0;
}
