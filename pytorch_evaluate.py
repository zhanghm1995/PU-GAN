'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-13 10:39:44
Email: haimingzhang@link.cuhk.edu.cn
Description: The evaluation code of Pytorch version.
'''

import os
import os.path as osp
import torch
import numpy as np
from glob import glob


def normalize_point_cloud(points):
    """Normalize point cloud to have zero mean and unit variance.

    Args:
        points (_type_): (B, N, 3)

    Returns:
        _type_: _description_
    """
    if isinstance(points, torch.Tensor):
        points_xyz = points[..., :3]
        centroid = torch.mean(points_xyz, axis=1, keepdims=True)
        furthest_distance = torch.max(
            torch.sqrt(torch.sum((points_xyz - centroid) ** 2, axis=-1)), dim=1, keepdims=True)[0]

        point_cloud_out = points.clone()
        point_cloud_out[..., :3] -= centroid
        point_cloud_out[..., :3] /= furthest_distance[..., None]
    else:
        centroid = np.mean(points[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((points[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)

        point_cloud_out = points.copy()
        point_cloud_out[..., :3] -= centroid
        point_cloud_out[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
    return point_cloud_out


def compute_cd_dist(pred, gt):
    from chamfer_distance import ChamferDistance as chamfer_dist

    dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
    dist = (torch.mean(dist1)) + (torch.mean(dist2))
    return dist


def evaluate_online(pred_dir, gt_dir, device):
    pred_files = sorted(glob(osp.join(pred_dir, "*.xyz")))
    gt_files = sorted(glob(osp.join(gt_dir, "*.xyz")))

    print(f"The length of pred_pc_list: {len(pred_files)}, gt_pc_list: {len(gt_files)}")

    pred_pc_list, gt_pc_list = [], []
    for pred_fp, gt_fp in zip(pred_files, gt_files):
        pred_pc = np.loadtxt(pred_fp)
        gt_pc = np.loadtxt(gt_fp)

        pred_pc_list.append(pred_pc)
        gt_pc_list.append(gt_pc)
    
    pred_pc_arr = torch.from_numpy(np.stack(pred_pc_list, axis=0)).to(torch.float32).to(device)
    gt_pc_arr = torch.from_numpy(np.stack(gt_pc_list, axis=0)).to(torch.float32).to(device)

    ## Normalize the point cloud
    pred = normalize_point_cloud(pred_pc_arr)
    gt = normalize_point_cloud(gt_pc_arr)

    # ## Compute the chamfer distance    
    cd_dist = compute_cd_dist(pred, gt)
    print("Mean CD dist: {:.8f}".format(cd_dist))


if __name__ == "__main__":
    pred_dir = "/home/zhanghm/Research/PU/PU-GAN/data/full_test/random_2048_output"
    gt_dir = "/data/data2/zhanghm/Datasets/PU/PU-GAN/test/gt_FPS_8192"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate_online(pred_dir, gt_dir, device)
