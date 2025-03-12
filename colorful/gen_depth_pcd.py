import argparse

import numpy as np

from PIL import Image

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import utility, data_io

def main(cfg):
    depth = Image.open(os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "bev_height.tiff"))
    depth = np.array(depth)
    # 获取所有符合步长的 u, v 坐标
    u_indices = np.arange(1, depth.shape[1], 4)
    v_indices = np.arange(1, depth.shape[0], 4)

    # 生成 u, v 网格
    uu, vv = np.meshgrid(u_indices, v_indices, indexing='ij')

    # 计算归一化的 u, v
    uu_scaled = uu / cfg["road_hdmapping_cam_scale"][0]
    vv_scaled = vv / cfg["road_hdmapping_cam_scale"][1]

    # 取出对应的 depth 值
    # depth_values = depth[uu, vv]
    depth_values = depth[vv, uu]

    # 使用 NumPy 拼接
    depth_pcd = np.column_stack((uu_scaled.ravel(), vv_scaled.ravel(), depth_values.ravel()))
    
    # bevcam2world = cfg["bevcam2ground_hdmapping"]
    # p = np.array([np.float64(x) for x in bevcam2world[0:3]])
    # q = [np.float64(x) for x in bevcam2world[3:7]]
    # q = utility.list2quat_xyzw(q)
    # depth_pcd = utility.transform_points(p, q, depth_pcd)
    depth_pcd_path = os.path.join(cfg["local_path"], cfg["road_hdmapping_subdir"], "depth.pcd")
    data_io.save_lidar(depth_pcd, depth_pcd_path, "pcd", "xyz")
    
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='visual lidar fusion')

    parser.add_argument('--config', type=str, default='', help='config name')
    args = parser.parse_args()
    cfg = utility.read_json(args.config)
    print(cfg)
    res = main(cfg=cfg)
    print("res ", res)
