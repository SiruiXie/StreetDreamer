import os
import argparse
from PIL import Image
import numpy as np
import cv2
from skimage.transform import resize
from luciddreamer import LucidDreamer
import warnings
from utils.graphics import fov2focal, focal2fov, getWorld2View, getProjectionMatrix
from scene.cameras import Camera, MiniCam
warnings.filterwarnings("ignore")
import json
import torch
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2

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


def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def apply_rotation(pose, rotation_matrix):

    R = pose[:3, :3]
    t = pose[:3, 3]

    new_R = rotation_matrix @ R
    return np.hstack((new_R, t.reshape(-1, 1)))


def rotate_cam_pose_left(pose, fovx):
    angle = -fovx
    left_rotation_matrix = rotation_matrix_y(angle)
    pose_left = apply_rotation(pose, left_rotation_matrix)
    return pose_left

def rotate_cam_pose_right(pose, fovx):
    angle = fovx
    left_rotation_matrix = rotation_matrix_y(angle)
    pose_left = apply_rotation(pose, left_rotation_matrix)
    return pose_left


def custom_transform(x, mu):
    assert 0 < mu < 1
    b = 0.8
    a = -b / (mu ** 2)
    
    return a * (x - mu) ** 2 + b
        

if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LucidDreamer')
    # Input options
    parser.add_argument('--ply_path', type=str, default='examples/Image015_animelakehouse.ply', help='generated ply file used for rendering')
    parser.add_argument('--back_dist', type=float, default=0, help='back distance for camera')
    parser.add_argument('--height', '-H', type=int, default=512, help='height of rendered image')
    parser.add_argument('--width', '-W', type=int, default=512, help='width of rendered image')
    parser.add_argument('--rgb_save_path', type=str, default='/data/xiesr/lucidsim/LucidSimNana/rgb_output.png', help='path to save rgb image')
    parser.add_argument('--depth_save_path', type=str, default='/data/xiesr/lucidsim/LucidSimNana/depth_output.png', help='path to save depth image')
    args = parser.parse_args()
    
    colormap = plt.get_cmap('jet')
    norm = colors.Normalize(vmin=0, vmax=1)
    
    ld = LucidDreamer(for_gradio=False, save_dir='render_results', use_tuned_inpainter=False, args=None, render_mode=True)
    ld.save_ply(fpath=args.ply_path)

    focal = [5.8269e+02, 5.8269e+02]
    FoVx = 2 * np.arctan(args.width / (2 * focal[0]))  # 根据新的宽度计算FoVx
    FoVy = 2 * np.arctan(args.height / (2 * focal[1]))  # 根据高度计算FoVy

    zfar = 100.0
    znear = 0.01

    c2w_0 = np.array([
        [1, 0, 0, -0],
        [0, -1, 0, -0],
        [0, 0, -1, -0 - args.back_dist]
    ]).astype(np.float32)
    
    c2w_left = rotate_cam_pose_left(c2w_0, FoVx)
    c2w_right = rotate_cam_pose_right(c2w_0, FoVx)
    
    c2w_left_left = rotate_cam_pose_left(c2w_left, FoVx)
    c2w_right_right = rotate_cam_pose_right(c2w_right, FoVx)
    
    all_c2ws = [c2w_left_left, c2w_left, c2w_0 , c2w_right, c2w_right_right]
    
    all_rgb = []
    all_depth = []
    for c2w in all_c2ws:

        c2w[:3, 1:3] *= -1
        if c2w.shape[0] == 3:
            one = np.zeros((1, 4))
            one[0, -1] = 1
            c2w = np.concatenate((c2w, one), axis=0)

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        w2c = torch.as_tensor(getWorld2View(R, T)).T.cuda()
        proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).T.cuda()

        cam = MiniCam(args.width, args.height, FoVx, FoVy, znear, zfar, w2c, w2c @ proj)
        rgb, depth = ld.render_img(H=args.height, W=args.width, pose=cam)
        all_rgb.append(rgb)
        all_depth.append(depth)
    
    
    rgb_img = np.hstack(all_rgb)
    depth_img = np.hstack(all_depth)[..., 0]

    # dpeth = (depth_img / depth_img.max() * 1.5)
    # dpeth = np.log(dpeth + 1) / np.log(1.8)
    # dpeth = custom_transform(depth_img)
    depth = (depth_img / depth_img.max())
    # depth = custom_transform(depth, mu=0.8)
    # depth = colormap(norm(depth))
    depth = (depth * 255).astype(np.uint8)
    # depth_color = cv2.applyColorMap(dpeth, cv2.COLORMAP_INFERNO)
    # depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    # pdb.set_trace()
    # depth = (depth[:, :, :3]).astype(np.uint8)

    Image.fromarray(rgb_img).save(args.rgb_save_path)
    Image.fromarray(depth).save(args.depth_save_path)
    # plt.imsave(args.depth_save_path, depth)


    
