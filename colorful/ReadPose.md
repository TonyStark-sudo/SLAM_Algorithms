### 读pose (timestamp x y z x y z w格式)
```c++
    std::vector<PoseStamped> image_poses;   
    std::ifstream infile(image_pose_path.string());
    if (!infile.is_open()) {
        std::cout << image_pose_path << " open failed" << std::endl;
        return;
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
        Pose pose_wc(position, rotation);
        Pose pose_mc = pose_mg * pose_gw * pose_wc;
        image_poses.emplace_back(std::stod(timestamp), timestamp, pose_mc);
    }
```