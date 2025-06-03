import os
import json
import glob
import re
import torch
import numpy as np  
import argparse
from torch import Tensor
from typing import Literal
import matplotlib.pyplot as plt


def load_rt_from_txt(file_path: str, comments: str = None) -> Tensor:
    return torch.from_numpy(np.loadtxt(file_path, comments=comments, dtype=np.float64))


def get_rt(folder: str) -> Tensor:
    files = sorted([x for x in glob(f"{folder}/*.txt") if re.search(r"(\d+)\.txt$", x)])
    return torch.stack([load_rt_from_txt(file) for file in files])


def rt34_to_44(rt: Tensor) -> Tensor:
    dummy = torch.tensor([[[0, 0, 0, 1]]] * rt.size(0), dtype=rt.dtype, device=rt.device)
    return torch.cat([rt, dummy], dim=1)


def relative_pose(rt: Tensor, mode: Literal["left", "right"]) -> Tensor:
    if mode == "left":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[:1].inverse() @ rt[1:]], dim=0)
    elif mode == "right":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[1:] @ rt[:1].inverse()], dim=0)
    return rt


def normalize_t(rt: Tensor, ref: Tensor = None, eps: float = 1e-9):
    if ref is None:
        ref = rt
    scale = ref[:, :3, 3:4].norm(p=2, dim=1).amax() + eps
    return rt34_to_44(torch.cat([rt[:, :3, :3], rt[:, :3, 3:4] / scale], dim=-1))

def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # N, 3, 3
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # N, 3
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # N, 3, 4
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)


def metric(c2w_1: Tensor, c2w_2: Tensor, mode: Literal["relative", "absolute"] = "relative") -> tuple[float, float, float]:  # N, 3, 4
    """gt: c2w_1, pred: c2w_2
    """
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()  # N, 3, 3



    if mode == "relative":

        c2w_1_rel = normalize_t(c2w_1, c2w_1)
        c2w_2_rel = normalize_t(c2w_2, c2w_2)

        # relative error
        TransErr = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()  # N, 3, 1
        CamMC = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()  # N, 3, 4
    
    elif mode == "absolute":

        c2w_1_abs = normalize_t(c2w_1, c2w_1)
        c2w_2_abs = normalize_t(c2w_2, c2w_1)
        # absolute error
        TransErr = calc_transerr(c2w_1_abs[:, :3, 3], c2w_2_abs[:, :3, 3]).sum().item()  # N, 3, 1
        CamMC = calc_cammc(c2w_1_abs[:, :3, :4], c2w_2_abs[:, :3, :4]).sum().item()  # N, 3, 4

    return RotErr, TransErr, CamMC


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix.
    Args:
        quaternion: (N, 4) tensor of quaternions in (x, y, z, w) format
    Returns:
        (N, 3, 3) rotation matrices
    """
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    x, y, z, w = quaternion.unbind(-1)
    
    rx = torch.stack((
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y)
    ), dim=-1)
    
    ry = torch.stack((
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x)
    ), dim=-1)
    
    rz = torch.stack((
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y)
    ), dim=-1)
    
    return torch.stack((rx, ry, rz), dim=-2)


def visualize_trajectories(gt_poses: Tensor, pred_poses: Tensor, save_path: str = None):
    """
    Visualize ground truth and predicted camera trajectories
    Args:
        gt_poses: (N, 8) ground truth poses (timestamp, tx, ty, tz, qx, qy, qz, qw)
        pred_poses: (N, 8) predicted poses (timestamp, tx, ty, tz, qx, qy, qz, qw)
        save_path: optional path to save the plot
    """
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions
    gt_positions = gt_poses[:, 1:4].numpy()    # [N, 3]
    pred_positions = pred_poses[:, 1:4].numpy() # [N, 3]
    
    # Plot trajectories
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
            'g-', label='Ground Truth', linewidth=2)
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
            'r--', label='Predicted', linewidth=2)
    
    # Plot camera positions
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], 
              c='green', marker='o', s=100, label='GT Start')
    ax.scatter(pred_positions[0, 0], pred_positions[0, 1], pred_positions[0, 2], 
              c='red', marker='o', s=100, label='Pred Start')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectories Comparison')
    
    # Add legend
    ax.legend()
    
    # Make axes equal
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    # set x[-1, 1], y[-1, 1], z[-1, 1]
    limits[:, 0] = -1
    limits[:, 1] = 1
    
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def downsample_trajectory(trajectory: Tensor, target_length: int) -> Tensor:
    """
    Downsample a trajectory to match target length using linear interpolation
    Args:
        trajectory: (N, 8) tensor of poses (timestamp, tx, ty, tz, qx, qy, qz, qw)
        target_length: desired number of frames
    Returns:
        (target_length, 8) downsampled trajectory
    """
    current_length = trajectory.shape[0]
    if current_length == target_length:
        return trajectory
    
    # Reshape for interpolation (batch_size=1, channels=8, length=N)
    trajectory_reshaped = trajectory.T.unsqueeze(0)  # [1, 8, N]
    
    # Use nn.functional.interpolate for all components at once
    interp_trajectory = torch.nn.functional.interpolate(
        trajectory_reshaped, 
        size=target_length,
        mode='linear',
        align_corners=True
    )  # [1, 8, target_length]
    
    # Reshape back to original format
    result = interp_trajectory.squeeze(0).T  # [target_length, 8]
    
    # Normalize quaternions (last 4 components)
    result[:, 4:8] = result[:, 4:8] / torch.norm(result[:, 4:8], dim=1, keepdim=True)
    
    return result



def evaluate(gt_path: str, pred_path: str, vis_path: str = None, aligment_way: Literal["truncate", "downsample"] = "truncate", mode: Literal["relative", "absolute"] = "relative"):
    # Load ground truth and predicted trajectories
    gt = load_rt_from_txt(gt_path)  # [N, 8] (timestamp, tx, ty, tz, qx, qy, qz, qw)
    pred = load_rt_from_txt(pred_path)  # [M, 8]
    
    # Downsample the longer trajectory to match the shorter one
    gt_len, pred_len = gt.shape[0], pred.shape[0]

    if aligment_way == "truncate":
        if gt_len > pred_len:
            # print(f"Truncating GT trajectory from {gt_len} to {pred_len} frames")
            gt = gt[:pred_len]
        elif pred_len > gt_len:
            # print(f"Truncating predicted trajectory from {pred_len} to {gt_len} frames")
            pred = pred[:gt_len]
    elif aligment_way == "downsample":
        if gt_len > pred_len:
            # print(f"Downsampling GT trajectory from {gt_len} to {pred_len} frames")
            gt = downsample_trajectory(gt, pred_len)
        elif pred_len > gt_len:
            # print(f"Downsampling predicted trajectory from {pred_len} to {gt_len} frames")
            pred = downsample_trajectory(pred, gt_len)
    
    # Visualize trajectories
    if vis_path:
        visualize_trajectories(gt, pred, vis_path)
    
    # Extract translations and quaternions
    gt_trans = gt[:, 1:4]  # [N, 3]
    gt_quat = gt[:, 4:8]   # [N, 4]
    
    pred_trans = pred[:, 1:4]  # [N, 3]
    pred_quat = pred[:, 4:8]   # [N, 4]

    
    # Convert quaternions to rotation matrices
    gt_rot = quaternion_to_matrix(gt_quat)    # [N, 3, 3]
    pred_rot = quaternion_to_matrix(pred_quat) # [N, 3, 3]
    
    # Combine into world-to-camera transforms
    gt_w2c = torch.cat([gt_rot, gt_trans.unsqueeze(-1)], dim=-1)    # [N, 3, 4]
    pred_w2c = torch.cat([pred_rot, pred_trans.unsqueeze(-1)], dim=-1) # [N, 3, 4]
    
    # Convert to camera-to-world matrices
    gt_c2w = rt34_to_44(gt_w2c).inverse()
    pred_c2w = rt34_to_44(pred_w2c).inverse()
    
    # Get relative poses
    gt_rel_c2w = relative_pose(gt_c2w, mode="left")
    pred_rel_c2w = relative_pose(pred_c2w, mode="left")
    
    # Calculate metrics
    metrics = ["RotErr", "TransErr", "CamMC"]
    items = metric(gt_rel_c2w.clone(), pred_rel_c2w.clone(), mode=mode)
    items = {metrics[i]: round(items[i], 3) for i in range(len(metrics))}
    
    # Print results
    # print(f"Mode: {mode}, Evaluation Results:")
    # print(items)
    
    return items


def batch_evaluate(gt_dir: str, pred_dir: str, aligment_way: Literal["truncate", "downsample"] = "truncate", mode: Literal["relative", "absolute"] = "relative", resolution: Literal["540p", "720p", "1080p",] = "720p"):
    results = []
    for directory in os.listdir(gt_dir):
        gt_path = os.path.join(gt_dir, directory, "pred_traj.txt")
        pred_path = os.path.join(pred_dir, "gen_" + resolution + "_" + directory, "pred_traj.txt")

        if os.path.exists(gt_path) and os.path.exists(pred_path):
            result = evaluate(gt_path, pred_path, aligment_way=aligment_way, mode=mode)
            result["dir"] = directory
            results.append(result)
        else:
            print(f"Skipping {directory} because one of the files does not exist")
            results.append({"dir": directory, "RotErr": None, "TransErr": None, "CamMC": None})

    return results

def summarize_results(results: list[tuple[float, float, float]], mode: Literal["relative", "absolute"] = "relative"):
    metrics = ["RotErr", "TransErr", "CamMC"]
    print(f"\n\nSummary of {mode} results:")
    for metric in metrics:
        print(f"{metric}: {np.mean([result[metric] for result in results if result[metric] is not None]):.3f}")

def save_results(results: list[tuple[float, float, float]], save_path: str = None):
    # save results to a json file
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth trajectory file or directory")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted trajectory file or directory")
    parser.add_argument("--vis_path", type=str, default="camera_trajectory.png", help="Path to save visualization plot")
    parser.add_argument("--aligment_way", type=str, default="truncate", help="Alignment way")
    parser.add_argument("--batch", action="store_true", help="Batch evaluation", default=False)
    parser.add_argument("--mode", type=str, default="relative", help="Evaluation mode")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--resolution", type=str, default="720p", help="Resolution")
    args = parser.parse_args()


    if args.batch:
        pred_dir = args.pred_path
        gt_dir = args.gt_path
        aligment_way = args.aligment_way
        mode = args.mode
        results = batch_evaluate(gt_dir, pred_dir, aligment_way, mode, args.resolution)
        summarize_results(results, mode)
        if args.save_path:
            save_results(results, args.save_path)
        else:
            print("No save path provided, results will not be saved")
    else:
        evaluate(args.gt_path, args.pred_path, args.vis_path, args.mode)
