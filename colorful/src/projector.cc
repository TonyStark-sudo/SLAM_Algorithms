#include "projector.h"
#include "pose.h"
#include "distribute_project.h"
#include "image_process.h"

#include <fstream>
#include <ostream>
#include <pcl/io/pcd_io.h>
#include <omp.h>

Projector::Projector(const std::string& config_path) : 
    color_map_(7, cv::Vec3b(0, 0, 0)), color_map_all_label_(66, cv::Vec3b(0, 0, 0)), label_map_(70, 0) {
    color_map_[0] = cv::Vec3b(0, 0, 0);
    color_map_[1] = cv::Vec3b(255, 255, 255);
    color_map_[2] = cv::Vec3b(170, 170, 170);
    color_map_[3] = cv::Vec3b(128, 64, 128);
    color_map_[4] = cv::Vec3b(244, 35, 232);
    color_map_[5] = cv::Vec3b(152, 251, 152);
    color_map_all_label_[0] = cv::Vec3b(0, 0, 0);
    color_map_all_label_[65] = cv::Vec3b(165, 42, 42); 
    color_map_all_label_[1] = cv::Vec3b(0, 192, 0);
    color_map_all_label_[2] = cv::Vec3b(196, 196, 196); 
    color_map_all_label_[3] = cv::Vec3b(190, 153, 153);
    color_map_all_label_[4] = cv::Vec3b(180, 165, 180);
    color_map_all_label_[5] = cv::Vec3b(90, 120, 150);
    color_map_all_label_[6] = cv::Vec3b(102, 102, 156);
    color_map_all_label_[7] = cv::Vec3b(128, 64, 255);
    color_map_all_label_[8] = cv::Vec3b(140, 140, 200);
    color_map_all_label_[9] = cv::Vec3b(170, 170, 170);
    color_map_all_label_[10] = cv::Vec3b(250, 170, 160);
    color_map_all_label_[11] = cv::Vec3b(96, 96, 96);
    color_map_all_label_[12] = cv::Vec3b(230, 150, 140);
    color_map_all_label_[13] = cv::Vec3b(128, 64, 128);
    color_map_all_label_[14] = cv::Vec3b(110, 110, 110);
    color_map_all_label_[15] = cv::Vec3b(244, 35, 232);
    color_map_all_label_[16] = cv::Vec3b(150, 100, 100);
    color_map_all_label_[17] = cv::Vec3b(70, 70, 70);
    color_map_all_label_[18] = cv::Vec3b(150, 120, 90);
    color_map_all_label_[19] = cv::Vec3b(220, 20, 60);
    color_map_all_label_[20] = cv::Vec3b(255, 0, 0);
    color_map_all_label_[21] = cv::Vec3b(255, 0, 100);
    color_map_all_label_[22] = cv::Vec3b(255, 0, 200);
    color_map_all_label_[23] = cv::Vec3b(200, 128, 128);
    color_map_all_label_[24] = cv::Vec3b(255, 255, 255);
    color_map_all_label_[25] = cv::Vec3b(64, 170, 64);
    color_map_all_label_[26] = cv::Vec3b(230, 160, 50);
    color_map_all_label_[27] = cv::Vec3b(70, 130, 180);
    color_map_all_label_[28] = cv::Vec3b(190, 255, 255);
    color_map_all_label_[29] = cv::Vec3b(152, 251, 152);
    color_map_all_label_[30] = cv::Vec3b(107, 142, 35);
    color_map_all_label_[31] = cv::Vec3b(0, 170, 30);
    color_map_all_label_[32] = cv::Vec3b(255, 255, 128);
    color_map_all_label_[33] = cv::Vec3b(250, 0, 30);
    color_map_all_label_[34] = cv::Vec3b(100, 140, 180);
    color_map_all_label_[35] = cv::Vec3b(220, 220, 220);
    color_map_all_label_[36] = cv::Vec3b(220, 128, 128);
    color_map_all_label_[37] = cv::Vec3b(222, 40, 40);
    color_map_all_label_[38] = cv::Vec3b(100, 170, 30);
    color_map_all_label_[39] = cv::Vec3b(40, 40, 40);
    color_map_all_label_[40] = cv::Vec3b(33, 33, 33);
    color_map_all_label_[41] = cv::Vec3b(100, 128, 160);
    color_map_all_label_[42] = cv::Vec3b(142, 0, 0);
    color_map_all_label_[43] = cv::Vec3b(70, 100, 150);
    color_map_all_label_[44] = cv::Vec3b(210, 170, 100);
    color_map_all_label_[45] = cv::Vec3b(153, 153, 153);
    color_map_all_label_[46] = cv::Vec3b(128, 128, 128);
    color_map_all_label_[47] = cv::Vec3b(0, 0, 80);
    color_map_all_label_[48] = cv::Vec3b(250, 170, 30);
    color_map_all_label_[49] = cv::Vec3b(192, 192, 192);
    color_map_all_label_[50] = cv::Vec3b(220, 220, 0);
    color_map_all_label_[51] = cv::Vec3b(140, 140, 20);
    color_map_all_label_[52] = cv::Vec3b(119, 11, 32);
    color_map_all_label_[53] = cv::Vec3b(150, 0, 255);
    color_map_all_label_[54] = cv::Vec3b(0, 60, 100);
    color_map_all_label_[55] = cv::Vec3b(0, 0, 142);
    color_map_all_label_[56] = cv::Vec3b(0, 0, 90);
    color_map_all_label_[57] = cv::Vec3b(0, 0, 230);
    color_map_all_label_[58] = cv::Vec3b(0, 80, 100);
    color_map_all_label_[59] = cv::Vec3b(128, 64, 64);
    color_map_all_label_[60] = cv::Vec3b(0, 0, 110);
    color_map_all_label_[61] = cv::Vec3b(0, 0, 70);
    color_map_all_label_[62] = cv::Vec3b(0, 0, 192);
    color_map_all_label_[63] = cv::Vec3b(32, 32, 32);
    color_map_all_label_[64] = cv::Vec3b(120, 10, 10);
    label_map_[7] = 1;
    label_map_[8] = 1;
    label_map_[23] = 1;
    label_map_[24] = 1;
    label_map_[2] = 2;
    label_map_[3] = 2;
    label_map_[4] = 2;
    label_map_[5] = 2;
    label_map_[6] = 2;
    label_map_[9] = 2;
    label_map_[13] = 3;
    label_map_[14] = 3;
    label_map_[41] = 3;
    label_map_[29] = 5;
    std::ifstream infile(config_path);
    if (infile.is_open()) {
        infile >> config_;
    }
    std::cout << "begin colorful" << std::endl;
}

// 将Matrix3d转换成数组，用于CUDA加速，mat3x3_to_pointer是私有成员函数
double* Projector::mat3x3_to_pointer(const Eigen::Matrix3d& Mat) {
    double* arr = new double[9];
    arr[0] = Mat(0, 0);
    arr[1] = Mat(0, 1);
    arr[2] = Mat(0, 2);

    arr[3] = Mat(1, 0);
    arr[4] = Mat(1, 1);
    arr[5] = Mat(1, 2);

    arr[6] = Mat(2, 0);
    arr[7] = Mat(2, 1);
    arr[8] = Mat(2, 2);

    return arr;
}

// 将Vector3d转换成数组，用于CUDA加速，vec3x1_to_pointer是私有成员函数
double* Projector::vec3x1_to_pointer(const Eigen::Vector3d& vec) {
    double* arr = new double[3];
    arr[0] = vec(0);
    arr[1] = vec(1);
    arr[2] = vec(2);

    return arr;
}

// 使用Brown-Conrady模型的畸变模型和针孔相机模型来进行投影
Eigen::Vector2d Projector::BrownConradyPhiholeProjection(const Eigen::Vector3d& Xcam,
                                                  const std::vector<double>& D,
                                                  const Eigen::Matrix3d& K) {
    // 相机系点的深度小于1.5m，认为是无效点
    if (Xcam.z() < 1.5) {
        return Eigen::Vector2d(-1, -1);
    }
    // 存储归一化平面的X,Y坐标
    Eigen::Vector2d Xp1(Xcam.x() / Xcam.z(), Xcam.y() / Xcam.z());

    double r = Xp1.norm();
    double k1 = D[0], k2 = D[1], p1 = D[2], p2 = D[3];

    // 计算畸变后的像素归一化坐标X、Y
    double xdist = Xp1(0) + Xp1(0) * (k1 * std::pow(r, 2) + k2 * std::pow(r, 4)) +
                 (p1 * (std::pow(r, 2) + 2 * std::pow(Xp1(0), 2)) +
                  2 * p2 * Xp1(0) * Xp1(1));
    double ydist = Xp1(1) + Xp1(1) * (k1 * std::pow(r, 2) + k2 * std::pow(r, 4)) +
                 (2 * p1 * Xp1(0) * Xp1(1) + p2 * (std::pow(r, 2) + 2 * std::pow(Xp1(1), 2)));
    Eigen::Vector2d Xp1d(xdist, ydist);

    // 得到畸变后的归一化坐标
    Eigen::Vector3d Xp1dh(Xp1d.x(), Xp1d.y(), 1.0);

    // 通过相机内参矩阵K得到像素坐标
    Eigen::Vector3d Xpix = K * Xp1dh;

    // 返回像素坐标（除以Xpix.z，确保是像素坐标）
    return Eigen::Vector2d(Xpix.x() / Xpix.z(), Xpix.y() / Xpix.z());
}

void Projector::ProjectColorful() {
    std::cout << "ProjectColorful" << std::endl;
    std::string cam_name = "cam_front_120";
    std::filesystem::path local_path = std::string(config_["local_path"]);
    // 读 intensity_rgb_ground 道路点云
    std::filesystem::path road_points_path = JoinPaths(local_path, std::string(config_["hdmapping_ground_subdir_fuser_lidar"]), "intensity_rgb_ground.pcd");
    // 读 lidar_mapping得到的相机位姿
    std::filesystem::path image_pose_path = JoinPaths(local_path, config_["lidar_mapping_save_subdir"], "pose_" + cam_name + "_optimized.txt");
    // 读取相机的raw数据和image_segmentation数据
    std::filesystem::path image_data_dir = JoinPaths(local_path, config_[cam_name + "_data_subdir"]);
    std::filesystem::path image_seg_dir = JoinPaths(local_path, config_[cam_name + "_img_seg_subdir"]);
    // 读resize后的相机内参
    std::filesystem::path cam_intrinsic = JoinPaths(local_path, config_[cam_name + "_new_intrinsic_subpath"]);
    std::cout << "road_points_path: " << road_points_path << std::endl;
    std::cout << "image_pose_path: " << image_pose_path << std::endl;
    std::cout << "image_data_dir: " << image_data_dir << std::endl;
    std::cout << "local_path: " << local_path << std::endl;
    std::cout << "cam_intrinsic: " << cam_intrinsic << std::endl;
    // 读世界坐标系到路面坐标系的外参
    Eigen::Vector3d p_gw = Eigen::Vector3d(config_["world2ground_hdmapping"][0], 
                                           config_["world2ground_hdmapping"][1], 
                                           config_["world2ground_hdmapping"][2]); 
    Eigen::Quaterniond q_gw(config_["world2ground_hdmapping"][6], config_["world2ground_hdmapping"][3], 
                            config_["world2ground_hdmapping"][4], config_["world2ground_hdmapping"][5]);
    Pose pose_gw(p_gw, q_gw);

    // 读bev相机坐标系到路面坐标系的外参
    Eigen::Quaterniond q_gm(config_["bevcam2ground_hdmapping"][6], config_["bevcam2ground_hdmapping"][3],
                            config_["bevcam2ground_hdmapping"][4], config_["bevcam2ground_hdmapping"][5]);
    Eigen::Vector3d p_gm(config_["bevcam2ground_hdmapping"][0], config_["bevcam2ground_hdmapping"][1],
                         config_["bevcam2ground_hdmapping"][2]);
    Pose pose_mg = Pose(p_gm, q_gm).Inverse();

    std::cout << "read poses" << std::endl;
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

    /*
    "road_hdmapping_cam_scale": [
        20.0,
        20.0
    ],
    */
    float scale_x = float(config_["road_hdmapping_cam_scale"][0]);
    float scale_y = float(config_["road_hdmapping_cam_scale"][1]);

    // pose_uv存储在bev坐标系下相机的2dpose
    // 每隔20m取一个pose_uv
    std::vector<Eigen::Vector2i> pose_uv;
    double step = 10.0;
    double odom = 0.0;
    double last_odom = -20.0;
    Eigen::Vector3d last_position;
    for (const PoseStamped& pose_stamped : image_poses) {
        Eigen::Vector3d position = pose_stamped.pose.p;
        if (last_odom < 0) {
            last_position = position;
            last_odom = 0;

            // 把在bev坐标系下的点的x、y坐标×scale_x、scale_y，得到在bev图像上的坐标
            int u = static_cast<int>(position.x() * scale_x);
            int v = static_cast<int>(position.y() * scale_y);
            pose_uv.emplace_back(u, v);
            continue;
        }
        odom += (position - last_position).norm();
        last_position = position;
        if (odom - last_odom > step) {
            int u = static_cast<int>(position.x() * scale_x);
            int v = static_cast<int>(position.y() * scale_y);
            pose_uv.emplace_back(u, v); 
            last_odom = odom;
        }
    }

    // 把pose_uv转换成数组uvs，用于CUDA加速
    int *uvs = new int[pose_uv.size() * 2];
    for (size_t i = 0; i < pose_uv.size(); i++) {
        uvs[2 * i] = pose_uv[i].x();
        uvs[2 * i + 1] = pose_uv[i].y();
    }
    /*
        "road_hdmapping_bev_xy_pixels": [
        8006,
        1810
    ],
    */
    int height = static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][1]);
    int width = static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][0]);

    // 读取降采样的的ground_fit点云
    std::filesystem::path point_cloud_for_height_path = JoinPaths(local_path, config_["vl_road"], "all_cam", "intensity_downsampled_ground_fit_m.pcd");
    // 将pcd点云读到point_cloud_for_height
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_for_height(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(point_cloud_for_height_path.string(), *point_cloud_for_height) == -1) {
        std::cout << point_cloud_for_height_path << " read failed" << std::endl;
        return;
    }
    std::cout << "begin depth image" << std::endl;
    ImageProcess depth_process;
    
    // 构造bev图上的点数组，转换成数组bev_points，用于CUDA加速
    float *bev_points = new float[3 * point_cloud_for_height->points.size()];
    /*
        "road_hdmapping_bev_z_range": [
        -1.7424628734588623,
        5.4158101081848145
    ],
        "road_hdmapping_bev_slack_z": 2.0,
    */
    float translation = double(config_["road_hdmapping_bev_z_range"][1]) + double(config_["road_hdmapping_bev_slack_z"]);
    int point_size = point_cloud_for_height->points.size();
    float *h_bev_height = new float[height * width];
    bool *h_bev_mask = new bool[height * width];
    for (size_t i = 0; i < point_size; ++i) {
        // 读降采样的ground_fit点云的point到bev_points数组，用于CUDA加速
        bev_points[3 * i] = point_cloud_for_height->points[i].x;
        bev_points[3 * i + 1] = point_cloud_for_height->points[i].y;
        bev_points[3 * i + 2] = point_cloud_for_height->points[i].z;
    }
    // 进行DepthImage计算，得到bev_height和bev_mask，size都是 height * width ???
    depth_process.DepthImage(bev_points, &point_size, &width, &height, &scale_x, &scale_y, h_bev_height, h_bev_mask, uvs, pose_uv.size());
    // 将bev_height所有元素初始化为config_["ground2bevcam_hdmapping"][2]
    cv::Mat bev_height = cv::Mat(height, width, CV_32FC1, cv::Scalar(config_["ground2bevcam_hdmapping"][2]));
    cv::Mat bev_mask = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat bev_test = cv::Mat::zeros(height, width, CV_8UC1);

    // 将CUDA计算得到的 h_bev_height 和 h_bev_mask 赋值给 bev_height 和 bev_mask
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = x + y * width;
            if (h_bev_mask[index]) {
                bev_height.at<float>(y, x) = h_bev_height[index];
                bev_mask.at<float>(y, x) = 1;
                bev_test.at<uchar>(y, x) = 255;
            }
        }
    }
    // 释放内存资源
    delete[] bev_points;
    delete[] h_bev_height;
    delete[] h_bev_mask;

    // 把bev_height和bev_mask写到到本地
    cv::imwrite(JoinPaths(local_path, config_["road_hdmapping_subdir"], "bev_height.tiff"), bev_height);
    cv::imwrite(JoinPaths(local_path, config_["road_hdmapping_subdir"], "bev_mask.tiff"), bev_mask);
    cv::imwrite(JoinPaths(local_path, config_["road_hdmapping_subdir"], "bev_mask.png"), bev_test);

    std::cout << "read intrinsic" << std::endl;
    nlohmann::json intrinsic;
    std::ifstream intrinsic_file(cam_intrinsic);
    if (intrinsic_file.is_open()) {
        intrinsic_file >> intrinsic;
    }

    Eigen::Matrix3d K;
    K << intrinsic["K"][0][0], intrinsic["K"][0][1], intrinsic["K"][0][2],
         intrinsic["K"][1][0], intrinsic["K"][1][1], intrinsic["K"][1][2],
         intrinsic["K"][2][0], intrinsic["K"][2][1], intrinsic["K"][2][2];
    double* h_K = mat3x3_to_pointer(K);
    std::vector<double> D {0.0, 0.0, 0.0};
    std::cout << "start read point cloud" << std::endl;

    // 读取读 intensity_rgb_ground 道路点云 intensity_rgb_ground.pcd
    pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_g(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(road_points_path.string(), *point_cloud_g) == -1) {
        std::cout << road_points_path << " read failed" << std::endl;
        return;
    }

    std::cout << "rgb" << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    point_cloud_rgb->resize(point_cloud_g->size());


    // 初始化一些host上的变量h_rgbs, h_points, h_counts, h_labels,用于CUDA计算
    size_t points_size = point_cloud_g->size();
    float* h_rgbs = new float[points_size * 3];
    float* h_points = new float[points_size * 3];
    float* h_counts = new float[points_size];
    int* h_labels = new int[points_size];
    for (size_t i = 0; i < points_size; ++i) {
        h_rgbs[3 * i] = 0;
        h_rgbs[3 * i + 1] = 0;
        h_rgbs[3 * i + 2] = 0;
        h_points[3 * i] = point_cloud_g->points[i].x;
        h_points[3 * i + 1] = point_cloud_g->points[i].y;
        h_points[3 * i + 2] = point_cloud_g->points[i].z;
        h_counts[i] = 0;
        h_labels[i] = 0;
    }

    // 初始化一个DistributeProjector对象， 用于CUDA投影
    DistributeProjector distribute_projector(h_points, h_rgbs, h_counts, points_size, label_map_, h_labels);
    float *points_m = new float[3 * distribute_projector.size_]; 
    double *rotation_mg = mat3x3_to_pointer(pose_mg.q.toRotationMatrix());
    double *translation_mg = vec3x1_to_pointer(pose_mg.p);
    std::cout << "transform" << std::endl;
    // 用于坐标系转换，将ground坐标系的点转到bev坐标系
    distribute_projector.TransformPointToM(rotation_mg, translation_mg, points_m);

    std::cout << "project" << std::endl;
    int total = image_poses.size();

    // 遍历每一个相机位姿，将相机拍摄的图像投影到bev图像上
    for (size_t i = 0; i < image_poses.size(); i++) {
        const auto& pose_mc = image_poses[i];
        std::cout << "\rProgress: " << std::setw(3) << i + 1 << "/" << total 
          << " [" << std::fixed << std::setprecision(1) 
          << (static_cast<double>(i + 1) / total) * 100 << "%]";
        std::cout.flush();
        std::filesystem::path image_path = JoinPaths(image_data_dir, pose_mc.timestamp_str + ".jpg");
        std::filesystem::path image_mask_path = JoinPaths(image_seg_dir, "mask", pose_mc.timestamp_str + ".png");
        std::filesystem::path label_seg_path = JoinPaths(image_seg_dir, "mask", pose_mc.timestamp_str + ".tiff");

        cv::Mat image = cv::imread(image_path.string());
        cv::Mat image_mask = cv::imread(image_mask_path.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_seg = cv::imread(label_seg_path, cv::IMREAD_UNCHANGED);

        unsigned char* h_image = new unsigned char[3 * image.cols * image.rows];
        unsigned char* h_image_mask = new unsigned char[image_mask.cols * image_mask.rows];
        unsigned char* h_image_seg = new unsigned char[image_seg.cols * image_seg.rows];
        for (size_t y = 0; y < image.rows; y++) {
            for (size_t x = 0; x < image.cols; x++) {
                h_image[3 * (y * image.cols + x)] = image.at<cv::Vec3b>(y, x)[0];
                h_image[3 * (y * image.cols + x) + 1] = image.at<cv::Vec3b>(y, x)[1];
                h_image[3 * (y * image.cols + x) + 2] = image.at<cv::Vec3b>(y, x)[2];
                h_image_mask[(y * image.cols + x)] = image_mask.at<unsigned char>(y, x);
                h_image_seg[y * image.cols + x] = static_cast<unsigned char>(image_seg.at<float>(y, x));
            }
        }
        // 得到从相机坐标系指向bev坐标系的旋转矩阵和平移向量
        Pose pose_cm = pose_mc.pose.Inverse();
        double* rotation = mat3x3_to_pointer(pose_cm.q.toRotationMatrix());
        double* translation = vec3x1_to_pointer(pose_cm.p);

        
        distribute_projector.Project(rotation, translation, h_image, h_image_mask, h_image_seg, h_K, image.rows, image.cols);
        delete[] rotation;
        delete[] translation;
        delete[] h_image;
        delete[] h_image_mask;
        delete[] h_image_seg;
    }
    // std::cout << std::endl <<  "end project" << std::endl;
    float* rgbs_res = new float[3 * distribute_projector.size_];
    int *label_res = new int[distribute_projector.size_];
    distribute_projector.Mix(rgbs_res, label_res);
    // std::cout << "mix color" << std::endl;
    // for (size_t i = 0; i < point_cloud_g->size(); i++) {
    //     point_cloud_rgb->points[i].x = points_m[3 * i];
    //     point_cloud_rgb->points[i].y = points_m[3 * i + 1];
    //     point_cloud_rgb->points[i].z = points_m[3 * i + 2];
    //     point_cloud_rgb->points[i].r = int(rgbs_res[3 * i]);
    //     point_cloud_rgb->points[i].g = int(rgbs_res[3 * i + 1]);
    //     point_cloud_rgb->points[i].b = int(rgbs_res[3 * i + 2]);
    // }
    std::cout << "generate bev_image" << std::endl;
    unsigned char *bev_image = new unsigned char[3 * width * height];
    unsigned char *bev_label = new unsigned char[width * height];
    memset(bev_image, 0, 3 * width * height * sizeof(unsigned char));
    memset(bev_label, 0, width * height * sizeof(unsigned char));
    distribute_projector.BevImage(&width, &height, &scale_x, &scale_y, bev_image, bev_label);
    cv::Mat image_bev = cv::Mat::zeros(static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][1]),
                                       static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][0]),
                                       CV_8UC3);
    cv::Mat label_bev = cv::Mat::zeros(static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][1]),
                                       static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][0]),
                                       CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image_bev.at<cv::Vec3b>(y, x) = cv::Vec3b(bev_image[3 * (x + y * width) + 2],
                                                      bev_image[3 * (x + y * width) + 1],
                                                      bev_image[3 * (x + y * width)]);
            label_bev.at<cv::Vec3b>(y, x) = color_map_[bev_label[x + y * width]];
        }
    }
    for (const Eigen::Vector2i& uv : pose_uv) {
        cv::circle(label_bev, cv::Point(uv.x(), uv.y()), 0, cv::Scalar(0, 0, 255), -1);
    }
    std::filesystem::path road_hdmapping_path = JoinPaths(local_path, std::string(config_["road_hdmapping_subdir"]));
    std::filesystem::create_directories(road_hdmapping_path);
    cv::imwrite(JoinPaths(local_path, road_hdmapping_path, "bev_image.png").string(), image_bev);
    cv::imwrite(JoinPaths(local_path, road_hdmapping_path, "bev_label_vis.png").string(), label_bev);

    std::cout << "project all" << std::endl;
    distribute_projector.ResetRGB();

    for (size_t i = 0; i < image_poses.size(); i++) {
        const auto& pose_mc = image_poses[i];
        std::cout << "\rProgress: " << std::setw(3) << i + 1 << "/" << total 
          << " [" << std::fixed << std::setprecision(1) 
          << (static_cast<double>(i + 1) / total) * 100 << "%]";
        std::cout.flush();

        // 得到相机的raw数据， 和 image_segmentation pipeline 的 mask 和 label的路径
        std::filesystem::path image_path = JoinPaths(image_data_dir, pose_mc.timestamp_str + ".jpg");
        std::filesystem::path image_mask_path = JoinPaths(image_seg_dir, "mask", pose_mc.timestamp_str + "_prelabel.png");
        std::filesystem::path label_seg_path = JoinPaths(image_seg_dir, "mask", pose_mc.timestamp_str + ".tiff");
        
        // 将raw数据和mask数据读成cv::Mat image, image_mask, image_seg
        cv::Mat image = cv::imread(image_path.string());
        cv::Mat image_mask = cv::imread(image_mask_path.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_seg = cv::imread(label_seg_path, cv::IMREAD_UNCHANGED);

        // 初始化一些host上的变量h_image, h_image_mask, h_image_seg，用于CUDA计算
        // h_image是3通道的RGB图像信息数组
        unsigned char* h_image = new unsigned char[3 * image.cols * image.rows];
        unsigned char* h_image_mask = new unsigned char[image_mask.cols * image_mask.rows];
        unsigned char* h_image_seg = new unsigned char[image_seg.cols * image_seg.rows];
        for (size_t y = 0; y < image.rows; y++) {
            for (size_t x = 0; x < image.cols; x++) {
                h_image[3 * (y * image.cols + x)] = image.at<cv::Vec3b>(y, x)[0];
                h_image[3 * (y * image.cols + x) + 1] = image.at<cv::Vec3b>(y, x)[1];
                h_image[3 * (y * image.cols + x) + 2] = image.at<cv::Vec3b>(y, x)[2];
                h_image_mask[(y * image.cols + x)] = image_mask.at<unsigned char>(y, x);

                // 强制转换成了uchar类型 image_seg和image_mask有什么区别？
                h_image_seg[y * image.cols + x] = static_cast<unsigned char>(image_seg.at<float>(y, x));
            }
        }

        // 得到相机坐标系指向bev坐标系的相机位姿
        Pose pose_cm = pose_mc.pose.Inverse();
        double* rotation = mat3x3_to_pointer(pose_cm.q.toRotationMatrix());
        double* translation = vec3x1_to_pointer(pose_cm.p);
        distribute_projector.ProjectNearest(rotation, translation, h_image, h_image_mask, h_image_seg, h_K, image.rows, image.cols);
        delete[] rotation;
        delete[] translation;
        delete[] h_image;
        delete[] h_image_mask;
        delete[] h_image_seg;
    }
    // std::cout << std::endl <<  "end project" << std::endl;
    distribute_projector.MixNearest(rgbs_res, label_res);
    // std::cout << "mix color" << std::endl;
    // for (size_t i = 0; i < point_cloud_g->size(); i++) {
    //     point_cloud_rgb->points[i].x = points_m[3 * i];
    //     point_cloud_rgb->points[i].y = points_m[3 * i + 1];
    //     point_cloud_rgb->points[i].z = points_m[3 * i + 2];
    //     point_cloud_rgb->points[i].r = int(rgbs_res[3 * i]);
    //     point_cloud_rgb->points[i].g = int(rgbs_res[3 * i + 1]);
    //     point_cloud_rgb->points[i].b = int(rgbs_res[3 * i + 2]);
    // }
    std::cout << "generate bev_image" << std::endl;
    memset(bev_image, 0, 3 * width * height * sizeof(unsigned char));
    unsigned char *bev_label_nearest = new unsigned char[width * height];
    memset(bev_label_nearest, 0, width * height * sizeof(unsigned char));
    distribute_projector.BevImageNearest(&width, &height, &scale_x, &scale_y, bev_image, bev_label_nearest);
    cv::Mat image_bev_nearest = cv::Mat::zeros(static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][1]),
                                       static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][0]),
                                       CV_8UC3);
    cv::Mat label_nearest_bev = cv::Mat::zeros(static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][1]),
                                        static_cast<int>(config_["road_hdmapping_bev_xy_pixels"][0]),
                                        CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image_bev_nearest.at<cv::Vec3b>(y, x) = cv::Vec3b(bev_image[3 * (x + y * width) + 2],
                                                      bev_image[3 * (x + y * width) + 1],
                                                      bev_image[3 * (x + y * width)]);
            label_nearest_bev.at<cv::Vec3b>(y,x) = color_map_all_label_[bev_label_nearest[x + y * width]];
        }
    }
    // for (const Eigen::Vector2i& uv : pose_uv) {
    //     cv::circle(label_nearest_bev, cv::Point(uv.x(), uv.y()), 0, cv::Scalar(0, 0, 255), -1);
    // }
    std::filesystem::create_directories(road_hdmapping_path);
    cv::imwrite(JoinPaths(local_path, road_hdmapping_path, "bev_image_prelabel.png").string(), image_bev_nearest);
    cv::imwrite(JoinPaths(local_path, road_hdmapping_path, "bev_label_prelabel_vis.png").string(), label_nearest_bev);
    // if (pcl::io::savePCDFileBinary(JoinPaths(local_path, road_hdmapping_path, "rgb.pcd").string(), *point_cloud_rgb) == -1) {
    //     std::cout << "write failed" << std::endl;
    // }

    delete[] h_K;
    delete[] h_rgbs;
    delete[] h_points;
    delete[] h_counts;
    delete[] h_labels;
    // delete[] rgbs_res;
    delete[] points_m;
    // delete[] label_res;
    // delete[] nearest_label_res;
}
