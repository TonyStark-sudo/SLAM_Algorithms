import argparse
import numpy as np
import cv2
from PIL import Image
from scipy.stats import norm as sci_norm
from pyquaternion import Quaternion
import multiprocessing as mp
import copy
import os
import sys
from tqdm import tqdm
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import utility, data_io
sys.path.append(str(Path(__file__).parent))
import gen_depth_pcd

def get_distance(a, b):
    diff = a - b
    dis = np.linalg.norm(diff)
    return dis

def process_intensity(intensities, method='CLAHE', clip_thresh=10.0, conf=0.85):
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)

    intensity_8bit = np.uint8(255.0 * (intensities - min_intensity) / (max_intensity - min_intensity))

    intensity_mat = np.reshape(intensity_8bit, (1, len(intensity_8bit)))

    if method == 'HIST_EQUALIZATION':
        processed_intensity = cv2.equalizeHist(intensity_mat)
    elif method == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit=clip_thresh, tileGridSize=(8, 8))
        processed_intensity = clahe.apply(intensity_mat)
    else:
        raise ValueError("Invalid method. Use 'HIST_EQUALIZATION' or 'CLAHE'.")
    processed_intensity = processed_intensity.reshape(-1)
    mean = np.mean(processed_intensity)
    std = np.std(processed_intensity)
    alpha = 1 - conf
    z = sci_norm.ppf(1 - alpha / 2)  
    lower_bound = mean - z * std
    upper_bound = mean + z * std
    clipped_intensity = np.clip(processed_intensity, lower_bound, upper_bound)
    scaled_intensity = (clipped_intensity - lower_bound) / (upper_bound - lower_bound) * 255
    return scaled_intensity

def render_label(image_size, label_path, prelabel=False):
    # print(label_path)
    label = np.load(label_path)
    label = cv2.resize(label, dsize=image_size, interpolation=cv2.INTER_NEAREST)
    mask_road = np.ones_like(label)
    label_off_road = ((0 <= label) & (label <= 2)) | (label == 3) | (label == 4) |(label == 5) | (label == 6) | ((9 <= label) & (label <= 12)) | ((15 <= label) & (label <= 22)) | ((25 <= label) & (label <= 29)) | ((30 <= label) & (label <= 40)) | (label >= 42)
    kernel_off_road = np.ones((4, 4), dtype=np.uint8)
    label_off_road = cv2.dilate(label_off_road.astype(np.uint8), kernel_off_road, 3).astype(np.bool_)

    label_movable = label >= 52
    kernel = np.ones((12, 12), dtype=np.uint8)
    label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(np.bool_)
    
    if prelabel:
        label_off_road = label_movable
    else:
        label_off_road = label_off_road | label_movable
    mask_road[label_off_road] = 0
    if prelabel:
        return mask_road

    _, image_h = image_size
    crop_cy = int(image_h * 0.4)

    label_ego = np.load(label_path)
    label_ego = cv2.resize(label_ego, dsize=image_size, interpolation=cv2.INTER_NEAREST)
    ego_seg = (label_ego == 64)

    mask_region = np.array(mask_road, dtype=np.uint8)
    num_regions, regions, _, _ = cv2.connectedComponentsWithStats(mask_region, connectivity=8)
    # region_count = np.bincount(regions.flatten())
    road_to_mask = []
    for i in range(1, num_regions):
        road_i = (regions == i).astype(np.uint8)
        road_dilated = cv2.dilate(road_i, kernel=np.ones((12, 12), np.uint8), iterations=1)
        intersect = np.logical_and(road_dilated, ego_seg)
        if np.any(intersect):
            # road_to_mask.append(road_i.astype(np.bool_))
            road_to_mask.append(cv2.dilate(road_i, kernel=np.ones((14, 14), np.uint8), iterations=4).astype(np.bool_))

    mask_road = np.zeros_like(mask_road, np.bool_)
    for road in road_to_mask:
        mask_road[road] = 1
    mask_road[label_movable] = 0
    mask_road[0:crop_cy, :] = 0
    mask_road = mask_road.astype(np.uint8)
    return mask_road

def gen_mask_worker(parma):
    cfg, camera_info, cam, timestamp, cam_mask_path = parma
    img_w, img_h = camera_info["img_size"]
    img_w, img_h = int(img_w), int(img_h)
    cam_img_seg_npy_dir = os.path.join(cfg["local_path"], cfg[f"{cam}_img_seg_npy_subdir"])
    sem_data_path = os.path.join(cam_img_seg_npy_dir, f"{timestamp}.npy")
    mask = render_label((img_w, img_h), sem_data_path)
    mask_prelabel = render_label((img_w, img_h), sem_data_path, prelabel=True)

    label = np.load(sem_data_path)
    label = cv2.resize(label, dsize=(img_w, img_h), interpolation=cv2.INTER_NEAREST)
    image = Image.fromarray(label)
    image.save(os.path.join(cam_mask_path, f"{timestamp}.tiff"))
    cv2.imwrite(os.path.join(cam_mask_path, f"{timestamp}.png"), mask * 255)
    cv2.imwrite(os.path.join(cam_mask_path, f"{timestamp}_prelabel.png"), mask_prelabel * 255)


def main(cfg, device):
    cfg["road_hdmapping_bev_resolution"] = 0.05
    cfg["road_hdmapping_bev_slack_z"] = 2.0
    cfg["road_hdmapping_cut_range"] = 16
    cfg["road_subdir"] = "road"
    road_dir = os.path.join(cfg["local_path"], cfg["road_subdir"])
    os.makedirs(road_dir, exist_ok=True)

    cfg["road_hdmapping_subdir"] = os.path.join(cfg["road_subdir"], "hdmapping")
    os.makedirs(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"]), exist_ok=True)

    cam = cfg["camera_names"][0]
    image_time = os.path.join(cfg["local_path"], cfg[f"{cam}_pose_cam_enu_subpath"])
    image_time = utility.get_txt_list(image_time)
    image_time = [line[0] for line in image_time]
    camera_info = utility.read_json(os.path.join(cfg["local_path"], cfg[f"{cam}_new_intrinsic_subpath"]))
    cam_mask_path = os.path.join(cfg["local_path"], cfg["cam_front_120_img_seg_subdir"], "mask")
    os.makedirs(cam_mask_path, exist_ok=True)
    params = []
    for timestamp in tqdm(image_time):
        param = [cfg, camera_info, cam, timestamp, cam_mask_path]
        params.append(param)
    proc_num = min(mp.cpu_count(), 10)
    proc_num = min(proc_num, len(params))
    if proc_num > 0:
        p = mp.Pool(proc_num)
        p.map(gen_mask_worker, params)
        p.close()
        p.join()

    lidar = cfg["lidar_names"][0]
    pil = np.array(cfg[f"{lidar}_to_ins"][:3])
    qil = np.array(cfg[f"{lidar}_to_ins"][3:])
    qil = Quaternion(qil[3], qil[0], qil[1], qil[2])
    qil = qil.normalised
    pli, qli = utility.inverse(pil, qil)
    cfg["road_hdmapping_intrinsic"] = os.path.join(cfg[f"road_hdmapping_subdir"], "intrinsic.json")
    cam_intrinsics = dict()
    for idx, cam in enumerate(cfg['road_camera_names']):
        cam_intrinsic = utility.read_json(os.path.join(cfg["local_path"], cfg[f"{cam}_new_intrinsic_subpath"]))
        cam_intrinsic["cam_id"] = idx
        cam_intrinsics[cam] = cam_intrinsic
    utility.write_json(cam_intrinsics, os.path.join(cfg["local_path"], cfg["road_hdmapping_intrinsic"]))

    lidar_poses = os.path.join(cfg["local_path"], cfg["lidar_mapping_save_subdir"], "pose_lidar_optimized.txt")
    lidar_poses = utility.get_txt_list(lidar_poses)
    lidar_poses = [utility.get_tum_pose_from_line(line) for line in lidar_poses]

    pgw = np.array(cfg["world2ground_hdmapping"][:3])
    qgw = Quaternion(cfg["world2ground_hdmapping"][6], *(cfg["world2ground_hdmapping"][3:6]))
    pwg, qwg = utility.inverse(pgw, qgw)

    xmin, ymin, zmin, xmax, ymax, zmax = cfg["ground_hdmapping_box"]

    xy_length = [xmax - xmin, ymax - ymin]
    cfg["road_hdmapping_bev_xy_pixels"] = [int(length / cfg["road_hdmapping_bev_resolution"] + 1) for length in xy_length]
    xy_length = [pixels * cfg["road_hdmapping_bev_resolution"] for pixels in cfg["road_hdmapping_bev_xy_pixels"]]
    cfg["road_hdmapping_bev_xy_length"] = xy_length

    cfg["road_hdmapping_bev_max_length"] = max(xy_length)
    cfg["road_hdmapping_bev_max_pixels"] = max(cfg["road_hdmapping_bev_xy_pixels"])
    cfg["road_hdmapping_bev_z_range"] = [zmin, zmax]

    bev_xy_center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
    xyz_offset = [bev_xy_center[i] - cfg["road_hdmapping_bev_max_length"] / 2 for i in range(2)]
    xyz_offset.append(0)

    pwb = qwg.rotate(np.array(xyz_offset)) + pwg
    qwb = qwg
    pbw, qbw = utility.inverse(pwb, qwb)
    cfg["bev2ground_hdmapping"] = xyz_offset
    cfg["world2bev_hdmapping"] = [pbw[0], pbw[1], pbw[2], qbw.x, qbw.y, qbw.z, qbw.w]

    bevcam_xy_top_left_corner = [bev_xy_center[0] - xy_length[0] / 2, bev_xy_center[1] + xy_length[1] / 2]
    bevcam_xy_top_left_corner.append(zmax + cfg["road_hdmapping_bev_slack_z"])

    qgm = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
    qgm = Quaternion._from_matrix(qgm)

    pgm = np.array(bevcam_xy_top_left_corner)

    pwm, qwm = utility.multiply(pwg, qwg, pgm, qgm)
    pmw, qmw = utility.inverse(pwm, qwm)

    pmg, qmg = utility.inverse(pgm, qgm)
    cfg["bevcam2ground_hdmapping"] = [pgm[0], pgm[1], pgm[2], qgm.x, qgm.y, qgm.z, qgm.w]
    cfg["ground2bevcam_hdmapping"] = [pmg[0], pmg[1], pmg[2], qmg.x, qmg.y, qmg.z, qmg.w]
    cfg["world2bevcam_hdmapping"] = [pmw[0], pmw[1], pmw[2], qmw.x, qmw.y, qmw.z, qmw.w]
    cfg["bevcam2world_hdmapping"] = [pwm[0], pwm[1], pwm[2], qwm.x, qwm.y, qwm.z, qwm.w]
    scale = [cfg["road_hdmapping_bev_xy_pixels"][i]/cfg["road_hdmapping_bev_xy_length"][i] for i in range(2)]
    cfg["road_hdmapping_cam_scale"] = scale

    intensity_path = os.path.join(cfg["local_path"], cfg[f"hdmapping_ground_subdir_{lidar}"], "intensity_ground.pcd")
    pointsi = data_io.read_lidar(intensity_path, "pcd", "xyzi")
    # pointsi = utility.transform_points(pmw, qmw, pointsi)
    pointsi = utility.transform_points(pmg, qmg, pointsi)
    pointsi[:,0] *= cfg["road_hdmapping_cam_scale"][0]
    pointsi[:,1] *= cfg["road_hdmapping_cam_scale"][1]
    processed_intensity = pointsi[:,3].reshape(-1)
    processed_intensity = process_intensity(processed_intensity)
    pointsi = np.concatenate((pointsi[:,:2], processed_intensity.reshape(-1,1)), axis=1)

    pts_pixel = pointsi[:,:2].astype(int)
    valid_pixel = (pts_pixel[:,0] >= 0) & (pts_pixel[:,0] < cfg["road_hdmapping_bev_xy_pixels"][0]) & (pts_pixel[:,1] >= 0) & (pts_pixel[:,1] < cfg["road_hdmapping_bev_xy_pixels"][1])
    pointsi = pointsi[valid_pixel]
    pts_pixel = pointsi[:,:2].astype(int)
    flat_indices = pts_pixel[:, 1] * cfg["road_hdmapping_bev_xy_pixels"][0] + pts_pixel[:, 0]
    img_intensity = np.zeros(cfg["road_hdmapping_bev_xy_pixels"][1] * cfg["road_hdmapping_bev_xy_pixels"][0])
    count = np.bincount(flat_indices)
    count2 = np.zeros(len(img_intensity) - len(count))
    count = np.concatenate((count, count2), axis=0)
    for i in range(len(flat_indices)):
        grid_index = flat_indices[i]
        img_intensity[grid_index] += processed_intensity[i]
    img_intensity = np.divide(img_intensity, count, out=np.zeros_like(img_intensity), where=count!=0)
    img_intensity = img_intensity.reshape(cfg["road_hdmapping_bev_xy_pixels"][1], cfg["road_hdmapping_bev_xy_pixels"][0])

    road_points_path = os.path.join(cfg["local_path"], cfg[f"vl_road"], "all_cam", "intensity_downsampled_ground_fit.pcd")
    pointsi_ground = data_io.read_lidar(road_points_path, "pcd", "xyzi")
    pgm = np.array(cfg["bevcam2ground_hdmapping"][:3])
    qgm = Quaternion(cfg["bevcam2ground_hdmapping"][6], *(cfg["bevcam2ground_hdmapping"][3:6]))
    pmg, qmg = utility.inverse(pgm, qgm)
    points_depth = utility.transform_points(pmg, qmg, pointsi_ground)
    road_points_path = os.path.join(cfg["local_path"], cfg[f"vl_road"], "all_cam", "intensity_downsampled_ground_fit_m.pcd")
    data_io.save_lidar(points_depth, road_points_path, "pcd", "xyzi")

    img_path = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg")
    gray_image = img_intensity.astype(np.uint8)
    cv2.imwrite(img_path, gray_image)
    utility.save_cfg(cfg, "hdmapping road main")
    path_file = str(Path(__file__).parent)
    command = f"{os.path.join(path_file, 'build', 'projector')} {os.path.join(cfg['local_path'], 'config.json')}"
    subprocess.run(command, shell=True)
    depth_mask = Image.open(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_mask.tiff"))
    depth_mask = np.array(depth_mask)
    depth_mask = depth_mask > 0
    bev_intensity = cv2.imread(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg"))
    bev_image = cv2.imread(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image.png"))
    bev_label = cv2.imread(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_label_vis.png"))
    bev_intensity[depth_mask == 0] = 0
    bev_image[depth_mask == 0] = 0
    bev_label[depth_mask == 0] = 0
    cv2.imwrite(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg"), bev_intensity)
    gray_image = cv2.imread(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg"), cv2.IMREAD_GRAYSCALE)
    rgba_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGBA)
    intensity_rgba_path = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_intensity.jpg")
    cv2.imwrite(intensity_rgba_path, rgba_image)
    cv2.imwrite(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image.png"), bev_image)
    cv2.imwrite(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_label_vis.png"), bev_label)
    lidar_trajectory = [[line[1][0], line[1][1], line[1][2]] for line in lidar_poses]
    lidar_trajectory = np.array(lidar_trajectory)
    lidar_trajectory = utility.transform_points(pmw, qmw, lidar_trajectory)

    lidar_trajectory[:,0] *= cfg["road_hdmapping_cam_scale"][0]
    lidar_trajectory[:,1] *= cfg["road_hdmapping_cam_scale"][1]
    bev_image = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image.png")
    bev_image = cv2.imread(bev_image)
    for pt in lidar_trajectory:
        cv2.circle(bev_image, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
    for i in range(len(lidar_trajectory)-1):
        start_point = lidar_trajectory[i][:2]
        end_point = lidar_trajectory[i + 1][:2]
        color = (255, 0, 0)  # 蓝色
        thickness = 1
        cv2.arrowedLine(bev_image, [int(start_point[0]),int(start_point[1])], [int(end_point[0]), int(end_point[1])], color, thickness, tipLength=0.15)
    cv2.imwrite(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_image_traj.png"), bev_image)

    bev_intensity = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_gray_image.jpg")
    bev_intensity = cv2.imread(bev_intensity)
    for pt in lidar_trajectory:
        cv2.circle(bev_intensity, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
    for i in range(len(lidar_trajectory)-1):
        start_point = lidar_trajectory[i][:2]
        end_point = lidar_trajectory[i + 1][:2]
        color = (255, 0, 0)  # 蓝色
        thickness = 1
        cv2.arrowedLine(bev_intensity, [int(start_point[0]),int(start_point[1])], [int(end_point[0]), int(end_point[1])], color, thickness, tipLength=0.15)
    cv2.imwrite(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_intensity_traj.png"), bev_intensity)
    # gen_depth_pcd.main(cfg)
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visual lidar fusion')

    parser.add_argument('--config', type=str, default='', help='config name')
    args = parser.parse_args()
    cfg = utility.read_json(args.config)
    print(cfg)
    res = main(cfg=cfg, device="cuda:0")
    print("res ", res)
