
import argparse
import os
import sys
import shutil
import json
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import scipy.sparse as sp
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import utility
import boto3
from datetime import datetime
import cv2
import math
import re

def quaternion_to_euler(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 超出范围时使用90度
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return roll, pitch, yaw

def get_distance_angle(a, b):
    dis = a[0] - b[0]
    dis = np.linalg.norm(dis)
    _, _, yawa = quaternion_to_euler(a[1])
    _, _, yawb = quaternion_to_euler(b[1])
    angle = yawa - yawb
    # angle = 2 * (math.acos((a[1].inverse * b[1]).w)) 
    # angle = math.degrees(angle)
    return dis, angle

def cut_map(cfg):
    lidar = cfg["lidar_names"][0]
    lidar_poses = os.path.join(cfg["local_path"], cfg["lidar_mapping_save_subdir"], "pose_lidar_optimized_key.txt")
    lidar_poses = utility.get_txt_list(lidar_poses)
    lidar_poses = [utility.get_tum_pose_from_line(line) for line in lidar_poses]
    pgm = np.array(cfg["bevcam2ground_hdmapping"][:3])
    qgm = Quaternion(cfg["bevcam2ground_hdmapping"][6], *(cfg["bevcam2ground_hdmapping"][3:6]))
    pmg, qmg = utility.inverse(pgm, qgm)
    pgw = np.array(cfg["world2ground_hdmapping"][:3])
    qgw = utility.list2quat_xyzw(cfg["world2ground_hdmapping"][3:])
    pmw, qmw = utility.multiply(pmg, qmg, pgw, qgw)

    pose_l2m_list = []
    for idx, frame in enumerate(lidar_poses):
        timestamp = frame[0]
        lidar_timestamp = timestamp
        p, q = frame[1], frame[2]
        lidar_pose_mat = utility.trans2mat(p, q)
        lidar_pose_mat = np.array(lidar_pose_mat).reshape(4, 4)
        lidar_pose_t = lidar_pose_mat[:3, -1]
        lidar_pose_r = lidar_pose_mat[:3, :3]
        lidar_pose_r = Quaternion._from_matrix(lidar_pose_r)
        pml, qml = utility.multiply(pmw, qmw, lidar_pose_t, lidar_pose_r)
        pose_l2m_mat = utility.trans2mat(pml, qml)
        pose_l2m_mat[0] *= cfg["road_hdmapping_cam_scale"][0]
        pose_l2m_mat[1] *= cfg["road_hdmapping_cam_scale"][1]
        pose_l2m_mat = np.array(pose_l2m_mat)
        pose_l2m_mat = pose_l2m_mat.reshape(-1)
        pose_l2m_mat = pose_l2m_mat.tolist()
        pose_l2m_mat = [timestamp] + pose_l2m_mat
        pose_l2m_list.append(pose_l2m_mat)
     
    ml_pose_path = os.path.join(cfg["local_path"], "data_post/pose_lidar2bev.txt")
    utility.write_txt(ml_pose_path, pose_l2m_list)

def main(cfg):
    camera_name = cfg["camera_names"][0]
    camera_info = utility.read_json(os.path.join(cfg["local_path"], cfg["road_hdmapping_intrinsic"]))
    camera_info = camera_info[camera_name]
    camera_K = camera_info["K"]
    camera_data_path = os.path.join(cfg["local_path"], cfg[f"{camera_name}_data_subdir"])

    data_post_path = os.path.join(cfg["local_path"], "data_post")
    if not os.path.exists(data_post_path):
        os.makedirs(data_post_path)
    data_post_camera_path = os.path.join(data_post_path, "images")
    if not os.path.exists(data_post_camera_path):
        os.makedirs(data_post_camera_path)
    now = datetime.now()
    formatted_time = now.strftime("ppl_bag_%Y%m%d_%H%M%S_0_450")
    folder_name = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"])

    bev_height_tiff = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_height.tiff")
    shutil.copyfile(bev_height_tiff, os.path.join(data_post_path, "bev_height.tiff"))
    bev_height = Image.open(bev_height_tiff)
    bev_height = np.array(bev_height)
    print("============ max min ===============")
    bev_height_max = float(np.max(bev_height))
    bev_height_min = float(np.min(bev_height))
    print("bev_height: ", bev_height_max, bev_height_min)
    # print(np.max(bev_height))
    # print(np.min(bev_height))
    sparse_matrix = sp.csr_matrix(bev_height)
    np.savez(os.path.join(data_post_path, f"bev_height.npz"), data=sparse_matrix.data, indices=sparse_matrix.indices, indptr=sparse_matrix.indptr,
             shape=sparse_matrix.shape)

    bev_image = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image.png")
    data_post_bev_image = os.path.join(data_post_path, "bev_image_no_traj.png")
    shutil.copyfile(bev_image, data_post_bev_image)

    mask_bev = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_mask.png")
    data_post_mask_bev = os.path.join(data_post_path, "bev_mask.png")
    shutil.copyfile(mask_bev, data_post_mask_bev)

    bev_image_traj = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image_traj.png")
    shutil.copyfile(bev_image_traj, os.path.join(data_post_path, "bev_image.png"))

    txt_paths = os.listdir(os.path.join(cfg["local_path"], cfg["lidar_mapping_save_subdir"]))
    pose_txt_dir = os.path.join(data_post_path, "pose_txt")
    os.makedirs(pose_txt_dir, exist_ok=True)
    for txt_path in txt_paths:
        if not txt_path.endswith(".txt"):
            continue
        pose_path = os.path.join(cfg["local_path"], cfg["lidar_mapping_save_subdir"], txt_path)
        shutil.copyfile(pose_path, os.path.join(pose_txt_dir, txt_path))

    nori_paths = os.listdir(os.path.join(cfg["local_path"], "data_preprocess"))
    # nori_paths = [m for m in nori_paths if re.search("nori", m)]
    nori_txt_dir = os.path.join(data_post_path, "nori_txt")
    os.makedirs(nori_txt_dir, exist_ok=True)
    for nori_path in nori_paths:
        if not nori_path.endswith(".txt"):
            continue
        npath = os.path.join(cfg["local_path"], "data_preprocess", nori_path)
        shutil.copyfile(npath, os.path.join(nori_txt_dir, nori_path))

    local_calib = os.path.join(cfg["local_path"], cfg["calib_subdir"])
    calib_dir = os.path.join(data_post_path, "calibresult")
    shutil.copytree(local_calib, calib_dir, dirs_exist_ok=True)

    # change
    pose_path = os.path.join(data_post_path,"pose_txt", f"pose_{camera_name}_optimized.txt")
    image_poses = utility.get_txt_list(pose_path)
    image_poses = [utility.get_tum_pose_from_line(line) for line in image_poses]

    bev_image_labe = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_label_vis.png")
    data_post_bev_label = os.path.join(data_post_path, "bev_label_vis.png")
    shutil.copyfile(bev_image_labe, data_post_bev_label)

    lidar = cfg["lidar_names"][0]
    intensity_pcd_path = os.path.join(cfg["local_path"], cfg[f"hdmapping_ground_subdir_{lidar}"], "intensity_downsampled.pcd")
    data_post_intensity_pcd = os.path.join(data_post_path, "intensity_downsampled.pcd")
    shutil.copyfile(intensity_pcd_path, data_post_intensity_pcd)

    # if (os.path.exists(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "depth.pcd"))):
    #     shutil.copyfile(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "depth.pcd"), os.path.join(data_post_path, "depth.pcd"))

    bev_intensity = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg")
    data_post_intensity_path = os.path.join(data_post_path, "bev_intensity_gray.jpg")
    shutil.copyfile(bev_intensity, data_post_intensity_path)
    
    gray_image = cv2.imread(data_post_intensity_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print(f"can't read: {data_post_intensity_path}")
        exit()
    rgba_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGBA)
    data_post_intensity_rgba_path = os.path.join(data_post_path, "bev_intensity.jpg")
    cv2.imwrite(data_post_intensity_rgba_path, rgba_image)

    config_path = os.path.join(cfg["local_path"], "config.json")
    cfg = utility.read_json(config_path)
    cfg["all_road_bev_xy_pixels"] = cfg["road_hdmapping_bev_xy_pixels"].copy()
    cfg["all_road_bev_max_length"] = cfg["road_hdmapping_bev_max_length"]
    cfg["all_road_bev_z_range"] = cfg["road_hdmapping_bev_z_range"].copy()
    cfg["road_bev_slack_z"] = cfg["road_hdmapping_bev_slack_z"]
    cfg["all_road_bev_xy_length"] = cfg["road_hdmapping_bev_xy_length"].copy()
    cfg["bevcam2world"] = cfg["bevcam2world_hdmapping"].copy()
    cfg["road_cam_scale"] = cfg["road_hdmapping_cam_scale"].copy()
    cfg["bev_height_range"] = [bev_height_min, bev_height_max]

    odom = 0.0
    orientation = 0.0
    last_pose = []
    odom_with_time = {}
    orientation_with_time = {}
    for i, line_cur in enumerate(image_poses):
        if len(last_pose) == 0:
            last_pose = line_cur[1:3]
            odom_with_time[line_cur[0]] = odom
            orientation_with_time[line_cur[0]] = orientation
            continue
        pose = line_cur[1:3]
        dis, angle = get_distance_angle(pose, last_pose)
        last_pose = pose
        odom += dis
        orientation += angle
        odom_with_time[line_cur[0]] = odom
        orientation_with_time[line_cur[0]] = orientation
        # print(orientation)
    camera_K_np = np.array(camera_K)
    pose = np.zeros((3, 4))
    pose[:3, :3] = np.eye(3)
    pose[0, 3] = 0
    pose[1, 3] = 0
    pose[2, 3] = 0
    transMatrix = camera_K_np @ pose

    txt_name = f"{camera_name}_pose_rogs.txt"
    time_pose_cam = []
    odom_step = 30
    odom_anno = -10000.0
    orientation_step = 30
    orientation_anno = 0.0
    print("all odom: ", odom)
    for i, img_pose in enumerate(image_poses):
        timestamp = img_pose[0]
        if not timestamp in odom_with_time:
            continue
        p, q = img_pose[1], img_pose[2]
        time_pose_cam.append([timestamp, p[0], p[1], p[2], q.x, q.y, q.z, q.w])
        if i != 0 and i + 1 != len(image_poses):
            if (odom - odom_with_time[timestamp] < 5.0) and (orientation - orientation_with_time[timestamp] < 5.0):
                continue
            elif (odom_with_time[timestamp] - odom_anno < odom_step and np.abs(orientation_with_time[timestamp] - orientation_anno) < orientation_step):
                continue

        print("odom: ", odom_with_time[timestamp])
        odom_anno = odom_with_time[timestamp]
        orientation_anno = orientation_with_time[timestamp]
        image_name = f"{timestamp}.jpg"
        image_name_out = f"{timestamp}.jpg"
        p, q = utility.inverse(p, q)
        pose_mat = utility.trans2mat(p, q)
        shutil.copyfile(os.path.join(camera_data_path, image_name), os.path.join(data_post_camera_path, image_name_out))

        json_dict = {}
        json_dict["poseMatrix"] = pose_mat.tolist()
        json_dict["transMatrix"] = transMatrix.tolist()
        json_name = f"{timestamp}.json"
        json_path = os.path.join(data_post_camera_path, json_name)
        with open(json_path, 'w') as jf:
            json.dump(json_dict, jf, indent=4)

    
    utility.write_txt(os.path.join(data_post_path, txt_name), time_pose_cam)

    all_camera_path = os.path.join(data_post_path, "all_images")
    os.makedirs(all_camera_path, exist_ok=True)
    
    for i, img_pose in enumerate(image_poses):
        timestamp = img_pose[0]
        if not timestamp in odom_with_time:
            continue
        p, q = img_pose[1], img_pose[2]
        time_pose_cam.append([timestamp, p[0], p[1], p[2], q.x, q.y, q.z, q.w])
        image_name = f"{timestamp}.jpg"
        image_name_out = f"{timestamp}.jpg"
        p, q = utility.inverse(p, q)
        pose_mat = utility.trans2mat(p, q)
        # 把图片cp到data_post的文件夹下
        # print(os.path.join(camera_data_path, image_name))
        shutil.copyfile(os.path.join(camera_data_path, image_name), os.path.join(all_camera_path, image_name_out))

        json_dict = {}
        json_dict["poseMatrix"] = pose_mat.tolist()
        json_dict["transMatrix"] = transMatrix.tolist()
        json_name = f"{timestamp}.json"
        json_path = os.path.join(all_camera_path, json_name)
        with open(json_path, 'w') as jf:
            json.dump(json_dict, jf, indent=4)
    prelabel_path = os.path.join(cfg["local_path"], "prelabel")
    if os.path.exists(prelabel_path):
        post_prelabel_path = os.path.join(data_post_path, "prelabel")
        if os.path.exists(post_prelabel_path):
            shutil.rmtree(post_prelabel_path)
        shutil.copytree(prelabel_path, post_prelabel_path)

    prelabel_v2_path = os.path.join(cfg["local_path"], "prelabel_v2")
    if os.path.exists(prelabel_v2_path):
        post_prelabel_v2_path = os.path.join(data_post_path, "prelabel_v2")
        if os.path.exists(post_prelabel_v2_path):
            shutil.rmtree(post_prelabel_v2_path)
        shutil.copytree(prelabel_v2_path, post_prelabel_v2_path)

    utility.save_cfg(cfg, "data_post")
    data_post_config_path = os.path.join(data_post_path, "config.json")
    utility.write_config(cfg, data_post_config_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data_post')
    parser.add_argument('--config', type=str, default='', help='config name')
    args = parser.parse_args()
    cfg = utility.read_json(args.config)
    print(cfg)
    res = main(cfg=cfg)
    print("res ", res)
