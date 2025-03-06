#include "iterativeClosestPoint.h"

void CreateICPProblem::find_feature_matches(std::vector<cv::KeyPoint> &kps_1, 
                                            std::vector<cv::KeyPoint> &kps_2, 
                                            std::vector<cv::DMatch> &matches) {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        cv::Mat des_1, des_2;
        detector->detect(img_1, kps_1);
        detector->detect(img_2, kps_2);
        descriptor->compute(img_1, kps_1, des_1);
        descriptor->compute(img_2, kps_2, des_2);

        // std::vector<cv::DMatch> match;
        // cv::DMatch 数据结构保存的是des_1和des_2的索引，以及两个描述子的距离
        matcher->match(des_1, des_2, matches);
        double min_dist = 10000, max_dist = 0;

        // 计算两个所有匹配好的描述子之间的最大和最小汉明距离
        for (auto &m : matches) {
            double dist = m.distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        // 采用距离小于 std::max(2 * min_dist, 30.0) 的过滤逻辑来计算成功匹配的点
        for (auto &m : matches) {
            if (m.distance <= std::max(2 * min_dist, 30.0)) {
                matches.push_back(m);
            }
        }
}

void CreateICPProblem::get_3D_points(std::vector<cv::Point3f>& points_1, std::vector<cv::Point3f>& points_2, 
                                     std::vector<cv::KeyPoint> &kps_1, std::vector<cv::KeyPoint> &kps_2, 
                                     std::vector<cv::DMatch> &matches) {

        cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
        for (const auto& m : matches) {
            int u1 = int(kps_1[m.queryIdx].pt.x);
            int v1 = int(kps_1[m.queryIdx].pt.y);
            int u2 = int(kps_2[m.trainIdx].pt.x);
            int v2 = int(kps_2[m.trainIdx].pt.y);

            ushort d1 = dpt_1.ptr<unsigned short>(v1)[u1];
            ushort d2 = dpt_2.ptr<unsigned short>(v2)[u2];
            if (d1 == 0 || d2 == 0) continue;

            float dd1 = float(d1) / 5000.0;
            float dd2 = float(d2) / 5000.0;
            cv::Point3d point_1, point_2;
            point_1.x = (u1 - K.at<double>(0, 2)) / K.at<double>(0, 0) * dd1;
            point_1.y = (v1 - K.at<double>(1, 2)) / K.at<double>(1, 1) * dd1;
            point_1.z = dd1;

            point_2.x = (u2 - K.at<double>(0, 2)) / K.at<double>(0, 0) * dd2;
            point_2.y = (v2 - K.at<double>(1, 2)) / K.at<double>(1, 1) * dd2;
            point_2.z = dd2;

            points_1.push_back(point_1);
            points_2.push_back(point_2);
        }
}


