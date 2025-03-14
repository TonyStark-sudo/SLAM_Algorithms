#include "utility.h"

std::vector<PoseStamped> readTumPoses(const std::string& pose_path) {
    std::vector<PoseStamped> poses;
    std::ifstream infile(pose_path);
    if (!infile.is_open()) {
        std::cout << pose_path << " is not opened !" << std::endl;
    }

    std::string line;
    while (getline(infile, line)) {
        std::stringstream ss(line);
        std::string timestamp;
        ss >> timestamp;
        std::string x;
        std::vector<double> line_vec;
        while (ss >> x) {
            line_vec.push_back(std::stod(x));
        }
        Eigen::Vector3d position(line_vec[0], line_vec[1], line_vec[2]);
        Eigen::Quaterniond rotation(line_vec[6], line_vec[3], line_vec[4], line_vec[5]);

        Pose pose(position, rotation);
        poses.emplace_back(std::stod(timestamp), timestamp, pose);
    }
    return poses; 
}

