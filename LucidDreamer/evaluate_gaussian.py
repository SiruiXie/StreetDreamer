import os
import argparse
from PIL import Image
import numpy as np
import cv2
from skimage.transform import resize
from luciddreamer import LucidDreamer
import warnings
from utils.trajectory import generate_seed_autodrive_render
from torchmetrics.multimodal import CLIPImageQualityAssessment
warnings.filterwarnings("ignore")
import json
from tqdm import tqdm
from loguru import logger
import pdb

def save_json(cam_paths, caption):
    js = {
        "camera_angle_x": 0.8279103882874479
    }
    frames = []
    for pose in cam_paths:
        frames.append({"transform_matrix": pose.tolist()})
    js["frames"] = frames
    with open(os.path.join('cameras', caption + '.json'), 'w', encoding="utf-8") as f:
        json.dump(js, f, indent=4, ensure_ascii=False)
    
metric = CLIPImageQualityAssessment(prompts=("quality", "natural", "real"))

if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LucidDreamer')
    # Input options
    parser.add_argument('--ply_path', type=str, default='examples/Image015_animelakehouse.ply', help='generated ply file used for rendering')
    parser.add_argument('--log_path', type=str, default='./render_log/log.txt', help='path to log file')
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    logger.add(args.log_path)
    logger.info(args)
    
    camera_poses = generate_seed_autodrive_render(n=60, num_offset=2, all_offset_length=0.2)
    yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
    render_poses = []
    for render_pose in camera_poses:
        Rw2i = render_pose[:3,:3]
        Tw2i = render_pose[:3,3:4]

        # Transfrom cam2 to world + change sign of yz axis
        Ri2w = np.matmul(yz_reverse, Rw2i).T
        Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
        Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
        # Pc2w = np.concatenate((Pc2w, np.array([0,0,0,1]).reshape((1,4))), axis=0)
        render_poses.append(Pc2w)
    # spdb.set_trace()
    new_camera_poses = np.stack(render_poses)
    
    save_json(new_camera_poses, caption='autodrive_render')


    ld = LucidDreamer(for_gradio=False, save_dir='render_results', use_tuned_inpainter=False, args=None, render_mode=True)
    ld.save_ply(fpath=args.ply_path)
    frames = ld.rennder_evaluation(preset='autodrive_render')
    metrics = metric(frames)
    
    for key, tensor in metrics.items():
        try:
            if tensor.numel() == 0:
                raise ValueError("张量为空，无法计算均值")
            mean_value = tensor.mean().item()
            logger.info(f"{key} 的均值是 {mean_value}")
        except Exception as e:
            logger.error(f"计算 {key} 的均值时出错：{e}")
    
    
    
    
    


