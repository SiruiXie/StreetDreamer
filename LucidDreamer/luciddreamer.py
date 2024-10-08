# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import glob
import torchvision
import json
import time
import datetime
import warnings
import shutil
from random import randint
from argparse import ArgumentParser
from loguru import logger
warnings.filterwarnings(action='ignore')
import sys
sys.path.insert(0, "/data/xiesr/LucidSim/LucidDreamer")

import pickle
import cv2
import imageio
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter

import torch
import torch.nn.functional as F
import gradio as gr
from diffusers import (
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, ControlNetModel, AutoPipelineForImage2Image)

from StableDiffusionControlNetInpaintingPipelineWrapper import Inpainter, ReInpainter
from IpAdapterWrapper import IpAdapterWrapper

from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.dataset_readers import loadCameraPreset
from utils.loss import l1_loss, ssim
from utils.camera import load_json
from utils.depth import colorize
from utils.lama import LaMa
from utils.trajectory import get_camerapaths, get_pcdGenPoses, generate_seed_autodrive_render


get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)

sd_model_path = '/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9'
controlnet_path = '/data/xiesr/lucidsim/LucidSim/Final_Eval/unet_full/checkpoint-controlnet-500'
inpaint_model_path = '/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots/afeee10def38be19995784bcc811882409d066e5'
unet_path = '/data/xiesr/lucidsim/LucidSim/Final_Eval/unet_full/checkpoint-unet-500'

only_tuned_controlnet_path = '/data/xiesr/lucidsim/LucidSim/controlnet-unet-decoder-warped_depth/checkpoint-controlnet-22500'
only_tuned_unet_path = '/data/xiesr/lucidsim/LucidSim/controlnet-unet-decoder-warped_depth/checkpoint-unet-22500'


def resize_and_center_crop(image, size=512):
    """
    将图像的短边缩放到512像素，并保持长边按比例缩小，然后进行512x512的中心裁剪。

    Args:
    image (PIL.Image.Image): 要处理的图像。
    size (int): 缩放后的短边大小和裁剪的最终大小，默认是512。

    Returns:
    PIL.Image.Image: 处理后的图像。
    """
    # 获取图像的宽度和高度
    width, height = image.size
    
    # 计算缩放比例
    if width < height:
        new_width = size
        new_height = int(height * (size / width))
    else:
        new_width = int(width * (size / height))
        new_height = size
    
    # 缩放图像
    resized_image = image.resize((new_width, new_height))
    
    # 计算中心裁剪区域
    left = (new_width - size) / 2
    top = (new_height - size) / 2
    right = (new_width + size) / 2
    bottom = (new_height + size) / 2
    
    # 裁剪图像
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image


def pointcloud_to_depth_map(point_cloud, rotation_mat, trans_vec, K, height, width):

    pts_coord_cam = rotation_mat.dot(point_cloud.T).T + trans_vec[:, 0]
    projected_points = np.matmul(K, pts_coord_cam.T).T

    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    depth_map = np.zeros((height, width))
    for i in tqdm(range(projected_points.shape[0]), desc='Projecting points to depth map'):
        u = int(np.round(projected_points[i, 0]))
        v = int(np.round(projected_points[i, 1]))

        if 0 <= u < width and 0 <= v < height:
            depth = pts_coord_cam[i, 2]
            if depth_map[v, u] == 0 or depth_map[v, u] > depth:
                depth_map[v, u] = depth

    return depth_map

def pointcloud_to_depth_map_and_valid_idx(point_cloud, rotation_mat, trans_vec, K, height, width):

    pts_coord_cam = rotation_mat.dot(point_cloud.T).T + trans_vec[:, 0]
    projected_points = np.matmul(K, pts_coord_cam.T).T

    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    depth_map = np.zeros((height, width))
    valid_idx = np.zeros((height, width)).astype(np.int32) - 1  # -1 for default
    for i in tqdm(range(projected_points.shape[0]), desc='Projecting points to depth map'):
        u = int(np.round(projected_points[i, 0]))
        v = int(np.round(projected_points[i, 1]))

        if 0 <= u < width and 0 <= v < height:
            depth = pts_coord_cam[i, 2]
            if depth_map[v, u] == 0 or depth_map[v, u] > depth:
                depth_map[v, u] = depth
                valid_idx[v, u] = i

    return depth_map, valid_idx

def write_to_pc_txt(pc, filename, color=None):
    assert len(pc.shape) == 2
    if pc.shape[0] == 3:
        new_pc = pc.T
    else:
        new_pc = pc
        
    if color is not None:
        assert len(color.shape) == 2
        if color.shape[0] == 3:
            new_color = color.T
        else:
            new_color = color
        assert new_color.shape[0] == new_pc.shape[0]
    

    f = open(filename, 'w')
    for i in range(new_pc.shape[0]):
        if color is None:
            f.write(f"{new_pc[i, 0]} {new_pc[i, 1]} {new_pc[i, 2]}\n")
        else:
            f.write(f"{new_pc[i, 0]} {new_pc[i, 1]} {new_pc[i, 2]} {new_color[i, 0]} {new_color[i, 1]} {new_color[i, 2]}\n")
    f.close()
    
class WrapperStableDiffusionInpaintPipeline:
    
    def __init__(self):
        self.model =  StableDiffusionInpaintPipeline.from_pretrained('/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9',
                                                                            revision='fp16', torch_type=torch.float16).to('cuda')
    def __call__(self,prompt: str, image: Image.Image, mask_image: Image.Image):
        img = self.model(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
        ).images[0]
    
        return mask_image, img



class LucidDreamer:
    def __init__(self, for_gradio=True, save_dir=None, use_tuned_inpainter=True, args=None, render_mode=False):
        self.args = args
        self.opt = GSParams()
        self.cam = CameraParams()
        bg_color = [1, 1, 1] if self.opt.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        self.save_dir = save_dir
        self.gaussians = GaussianModel(self.opt.sh_degree)
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        logger.add(f"{self.save_dir}/log.txt")
        
        if not render_mode:
            self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_K', source='local', pretrained=True, trust_repo=True, skip_validation=True,).to('cuda')
            # if args.use_naive_sd:
            self.rgb_model = WrapperStableDiffusionInpaintPipeline()
            # self.reinpainter = None
            # elif args.tuned_controlnet:
            
            # self.rgb_model = Inpainter(controlnet_path, unet_path, sd_model_path, device='cuda')
            #     self.reinpainter = ReInpainter(self.d_model, only_tuned_controlnet_path, only_tuned_unet_path, sd_model_path, noise_steps=100, ).to('cuda')
            
            # if args.personalized:
            # self.rgb_model = Inpainter("/data/xiesr/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-depth/snapshots/35e42a3ea49845b3c76f202f145f257b9fb1b7d4", 
            #                            "/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/unet", 
            #                            sd_model_path, device='cuda')
            # self.reinpainter = ReInpainter(self.d_model, controlnet_path=controlnet_path, unet_model_path=unet_path, sd_model_path=sd_model_path, noise_steps=100, ).to('cuda')
            # self.reinpainter = ReInpainter(self.d_model, controlnet_path, unet_path, sd_model_path, noise_steps=100, ).to('cuda')
            self.reinpainter = ReInpainter(self.d_model, "/data/xiesr/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-depth/snapshots/35e42a3ea49845b3c76f202f145f257b9fb1b7d4", 
                                           "/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/unet", 
                                           sd_model_path, noise_steps=100, ).to('cuda')
            
            # self.rgb_model = Inpainter(controlnet_path, unet_path, sd_model_path, device='cuda')
            # self.reinpainter = ReInpainter(self.d_model, controlnet_path, unet_path, sd_model_path, noise_steps=100, ).to('cuda')
            
            # elif args.ip_adapter:
            # self.rgb_model = IpAdapterWrapper(control_model_path="lllyasviel/control_v11f1p_sd15_depth", sd_model_path=sd_model_path, device='cuda')
            # self.reinpainter = None
        
            self.for_gradio = for_gradio
            self.root = 'outputs'
            self.default_model = 'SD1.5 (default)'
            
            
            # if (not use_tuned_inpainter):
            #     self.rgb_model = StableDiffusionInpaintPipeline.from_pretrained(
            #     #     'runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16).to('cuda')
            #         '/data/xiesr/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9', revision='fp16', torch_dtype=torch.float16).to('cuda')
            # else:
            #     self.rgb_model = Inpainter(only_tuned_controlnet_path, only_tuned_unet_path, sd_model_path, device='cuda') #.to('cuda')
            #     self.reinpainter = ReInpainter(self.d_model, only_tuned_controlnet_path, only_tuned_unet_path, sd_model_path, noise_steps=100, ).to('cuda')
            # # self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
            # self.d_model = torch.hub.load("/data/xiesr/.cache/torch/hub/isl-org_ZoeDepth_main")
            self.controlnet = None
            self.lama = None
            self.current_model = self.default_model

    def load_model(self, model_name, use_lama=True):
        if model_name is None:
            model_name = self.default_model
        if self.current_model == model_name:
            return
        if model_name == self.default_model:
            self.controlnet = None
            self.lama = None
            self.rgb_model = Inpainter(controlnet_path, unet_path, inpaint_model_path, sd_model_path).to('cuda')
            self.reinpainter = ReInpainter(self.d_model, controlnet_path, unet_path, sd_model_path, noise_steps=20).to('cuda')
        else:
            if self.controlnet is None:
                self.controlnet = ControlNetModel.from_pretrained(
                    'lllyasviel/control_v11p_sd15_inpaint', torch_dtype=torch.float16)
            if self.lama is None and use_lama:
                self.lama = LaMa('cuda')
            self.rgb_model = Inpainter(controlnet_path, unet_path, inpaint_model_path, sd_model_path).to('cuda')
            self.reinpainter = ReInpainter(self.d_model, controlnet_path, unet_path, sd_model_path, noise_steps=200)
            # self.rgb_model.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        self.current_model = model_name

    # def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
    #     image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
    #     mask_pil = Image.fromarray(np.round((1 - mask_image) * 255.).astype(np.uint8))
    #     if self.current_model == self.default_model:
    #         return self.rgb_model(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             generator=generator,
    #             num_inference_steps=num_inference_steps,
    #             image=image_pil,
    #             mask_image=mask_pil,
    #         ).images[0]

    #     kwargs = {
    #         'negative_prompt': negative_prompt,
    #         'generator': generator,
    #         'strength': 0.9,
    #         'num_inference_steps': num_inference_steps,
    #         'height': self.cam.H,
    #         'width': self.cam.W,
    #     }

    #     image_np = np.round(np.clip(image, 0, 1) * 255.).astype(np.uint8)
    #     mask_sum = np.clip((image.prod(axis=-1) == 0) + (1 - mask_image), 0, 1)
    #     mask_padded = pad_mask(mask_sum, 3)
    #     masked = image_np * np.logical_not(mask_padded[..., None])

    #     if self.lama is not None:
    #         lama_image = Image.fromarray(self.lama(masked, mask_padded).astype(np.uint8))
    #     else:
    #         lama_image = image

    #     mask_image = Image.fromarray(mask_padded.astype(np.uint8) * 255)
    #     control_image = self.make_controlnet_inpaint_condition(lama_image, mask_image)

    #     return self.rgb_model(
    #         prompt=prompt,
    #         image=lama_image,
    #         control_image=control_image,
    #         mask_image=mask_image,
    #         **kwargs,
    #     ).images[0]
    
    def rgb(self, image, control_image, mask_image):
        if isinstance(self.rgb_model, Inpainter):
            return self.rgb_model(
                image=image,
                control_image=control_image,
                mask_image=mask_image,
            )
        elif isinstance(self.rgb_model, WrapperStableDiffusionInpaintPipeline):
            return self.rgb_model(
                prompt='',
                image=image,
                mask_image=mask_image,
            )
        elif isinstance(self.rgb_model, IpAdapterWrapper):
            return self.rgb_model(
                prompt='A peaceful suburban street lined with cars and modest homes, casting long shadows.',
                adaper_image_or_url=image,
                mask_pil=mask_image,
                depth_img_or_url=control_image,
            )

    def d(self, im):
        return self.d_model.infer_pil(im)

    def make_controlnet_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def run(self, rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, render_camerapath, model_name=None, example_name=None):
        gaussians = self.create(
            rgb_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, model_name, example_name)
        gallery, depth = self.render_video(render_camerapath, example_name=example_name)
        return (gaussians, gallery, depth)

    def create(self, rgb_cond, depth_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, model_name=None, example_name=None, args=None, save_to="tmp.npy", load_from=False):

        if self.for_gradio:
            self.cleaner()
            self.load_model(model_name)
        if example_name and example_name != 'DON\'T':
            outfile = os.path.join('examples', f'{example_name}.ply')
            if not os.path.exists(outfile):
                if load_from:
                    self.traindata = np.load(os.path.join(self.save_dir, save_to), allow_pickle=True).item()
                else:
                    self.traindata = self.generate_pcd(rgb_cond, depth_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, args=args)
                    if save_to:
                        np.save(os.path.join(self.save_dir, save_to), self.traindata)
                self.scene = Scene(self.traindata, self.gaussians, self.opt)        
                self.training()
            outfile = self.save_ply(outfile)
        else:
            if load_from:
                self.traindata = np.load(os.path.join(self.save_dir, save_to), allow_pickle=True).item()
            else:
                self.traindata = self.generate_pcd(rgb_cond, depth_cond, txt_cond, neg_txt_cond, pcdgenpath, seed, diff_steps, args=args)
                if save_to:
                    np.save(os.path.join(self.save_dir, save_to), self.traindata)
            self.scene = Scene(self.traindata, self.gaussians, self.opt)        
            self.training()
            self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            outfile = self.save_ply(os.path.join(self.save_dir, f'{str(self.timestamp)}_gsplat.ply'))
        return outfile
    
    def save_ply(self, fpath=None):
        if fpath is None:
            dpath = os.path.join(self.root, self.timestamp)
            fpath = os.path.join(dpath, 'gsplat.ply')
            os.makedirs(dpath, exist_ok=True)
        if not os.path.exists(fpath):
            self.gaussians.save_ply(fpath)
        else:
            self.gaussians.load_ply(fpath)
        return fpath

    def cleaner(self):
        # Remove the temporary file created yesterday.
        for dpath in glob.glob(os.path.join('/tmp/gradio', '*',  self.root, '*')):
            timestamp = datetime.datetime.strptime(os.path.basename(dpath), '%y%m%d_%H%M%S')
            if timestamp < datetime.datetime.now() - datetime.timedelta(days=1):
                try:
                    shutil.rmtree(dpath)
                except OSError as e:# self.scene.getPresetCameras(self.args.campath_render)
                    print("Error: %s - %s." % (e.filename, e.strerror))
        if self.for_gradio:
            # Delete gsplat.ply if exists
            if os.path.exists('./gsplat.ply'):
                os.remove('./gsplat.ply')
    
    def render_img(self, H, W, pose):
        self.cam = CameraParams(H, W)
        results = render(pose, self.gaussians, self.opt, self.background)
        rgb, depth = results['render'], results['depth']
        rgb = np.round(rgb.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8)
        depth = depth.permute(1,2,0).detach().cpu().numpy()
        # depth = Image.fromarray(depth)
        # rgb = Image.fromarray(rgb)
        self.cam = CameraParams()

        return rgb, depth
    
    def rennder_evaluation(self, preset, example_name=None, progress=gr.Progress()):
        if not hasattr(self, 'scene'):
            views = load_json(os.path.join('cameras', f'{preset}.json'), self.cam.H, self.cam.W)
            # views = generate_seed_autodrive_render(n=30, num_offset=6, all_offset_length=4.8)
        else:
            views = self.scene.getPresetCameras(preset)
        
        iterable_render = tqdm(views, desc='[4/4] Rendering for evaluation')
        framelist = []
        
        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame, _ = results['render'], results['depth']
            # framelist.append(
            #     np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            framelist.append(
                (frame.detach().cpu().clip(0, 1) * 255.).float()
            )
        frames = torch.stack(framelist)
        return frames
            # depth = -(depth * (depth > 0)).detach().cpu().numpy()
            # dmin_local = depth.min().item()
            # dmax_local = depth.max().item()
            # if dmin_local < dmin:
            #     dmin = dmin_local
            # if dmax_local > dmax:
            #     dmax = dmax_local
            # depthlist.append(depth)
        
        
        

    def render_video(self, preset, example_name=None, progress=gr.Progress()):
        if example_name and example_name != 'DON\'T':
            videopath = os.path.join('examples', f'{example_name}_{preset}.mp4')
            depthpath = os.path.join('examples', f'depth_{example_name}_{preset}.mp4')
        else:
            if self.for_gradio:
                os.makedirs(os.path.join(self.root, self.timestamp), exist_ok=True)
                videopath = os.path.join(self.root, self.timestamp, f'{preset}.mp4')
                depthpath = os.path.join(self.root, self.timestamp, f'depth_{preset}.mp4')
            else:
                videopath = os.path.join(self.save_dir, f'{preset}.mp4')
                depthpath = os.path.join(self.save_dir, f'depth_{preset}.mp4')
        if os.path.exists(videopath) and os.path.exists(depthpath):
            os.remove(videopath)
            os.remove(depthpath)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)    
        # return videopath, depthpath
        
        if not hasattr(self, 'scene'):
            views = load_json(os.path.join('cameras', f'{preset}.json'), self.cam.H, self.cam.W)
            # views = generate_seed_autodrive_render(n=30, num_offset=6, all_offset_length=4.8)
        else:
            views = self.scene.getPresetCameras(preset)
        
        # import IPython; IPython.embed()
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8

        if self.for_gradio:
            iterable_render = progress.tqdm(views, desc='[4/4] Rendering a video')
        else:
            iterable_render = tqdm(views, desc='[4/4] Rendering a video')

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth']
            framelist.append(
                np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depth = -(depth * (depth > 0)).detach().cpu().numpy()
            dmin_local = depth.min().item()
            dmax_local = depth.max().item()
            if dmin_local < dmin:
                dmin = dmin_local
            if dmax_local > dmax:
                dmax = dmax_local
            depthlist.append(depth)

        progress(1, desc='[4/4] Rendering a video...')

        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        # if not os.path.exists(videopath):
        imageio.mimwrite(videopath, framelist, fps=40, quality=8)
        # if not os.path.exists(depthpath):
        imageio.mimwrite(depthpath, depthlist, fps=40, quality=8)
        return videopath, depthpath

    def training(self, progress=gr.Progress()):
        if not self.scene:
            raise('Build 3D Scene First!')
        
        # breakpoint()
        
        if self.for_gradio:
            iterable_gauss = progress.tqdm(range(1, self.opt.iterations + 1), desc='[3/4] Baking Gaussians')
        else:
            iterable_gauss = tqdm(list(range(1, self.opt.iterations + 1)))

        for iteration in iterable_gauss:
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration == 100 or iteration == 300 or iteration == 500:
                self.gaussians.oneupSHdegree()

            if iteration <= 1000:
                # Pick a random Camera
                viewpoint_stack = self.scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

                # import pdb; pdb.set_trace()
                # Render
                render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
                image, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                loss.backward()
            
            else:
                views = self.traindata['frames']
                random_idx = randint(0, len(views)-1)
                train_camera = [x for x in self.scene.train_cameras if x.colmap_id == random_idx][0]
                train_depth = torch.tensor(views[random_idx]['depth']).to('cuda')
                train_depth = train_depth / train_depth.max()
                # add gaussian blur to train_depth
                train_depth2 = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=2)(train_depth[None, ...])
                #! TODO
                
                render_pkg = render(train_camera, self.gaussians, self.opt, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth = (
                    render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'], render_pkg['depth'])
                
                gt_image = train_camera.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = ((1.0 - self.opt.lambda_dssim * 3) * Ll1 + self.opt.lambda_dssim * 3 * (1.0 - ssim(image, gt_image))) * 0.2
                
                render_img, render_mask = image, depth > 0
                loss += self.reinpainter.train_step(render_img, render_mask, 50, 200, depth=train_depth2)
                loss.backward()
                torchvision.utils.save_image(image.cpu(), f"{iteration}.png")
                torchvision.utils.save_image(train_depth2.cpu().float(), f"{iteration}_1.png")

            with torch.no_grad():
                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

    def generate_pcd(self, rgb_cond, depth_cond, prompt, negative_prompt, pcdgenpath, seed, diff_steps, progress=gr.Progress(), args=None):
        assert args is not None
        ## processing inputs
        generator=torch.Generator(device='cuda').manual_seed(seed)
        
        rgb_cond = resize_and_center_crop(rgb_cond)

        w_in, h_in = rgb_cond.size  # code here needs to be changed
        if w_in/h_in > 1.1 or h_in/w_in > 1.1: # if height and width are similar, do center crop
            in_res = max(w_in, h_in)
            # create a blank image
            image_in, mask_in = np.zeros((in_res, in_res, 3), dtype=np.uint8), 255*np.ones((in_res, in_res, 3), dtype=np.uint8)
            depth_in = np.zeros((in_res, in_res, 3), dtype=np.float32)
            depth_image_in = self.d(rgb_cond)
            temp_depth = np.array(depth_image_in)[..., None].repeat(3, axis=-1)
            temp_depth = temp_depth / temp_depth.max() * 255.
            
            # fill in the image and mask right in the middle
            image_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = np.array(rgb_cond)
            mask_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = 0
            depth_in[int(in_res/2-h_in/2):int(in_res/2+h_in/2), int(in_res/2-w_in/2):int(in_res/2+w_in/2)] = temp_depth
            depth_in = depth_in.astype(np.uint8)

            image2 = Image.fromarray(image_in).resize((self.cam.W, self.cam.H))
            mask2 = Image.fromarray(mask_in).resize((self.cam.W, self.cam.H))
            depth2 = Image.fromarray(depth_in).resize((self.cam.W, self.cam.H))
            # image_curr = self.rgb(
            #     prompt=prompt,
            #     image=image2,
            #     negative_prompt=negative_prompt, generator=generator,
            #     mask_image=mask2,
            # )
            image_curr = self.rgb(
                image = image2,
                control_image = depth2,
                mask_image=mask2
            )

        else: # if there is a large gap between height and width, do inpainting
            if w_in > h_in:
                image_curr = rgb_cond.crop((int(w_in/2-h_in/2), 0, int(w_in/2+h_in/2), h_in)).resize((self.cam.W, self.cam.H))
            else: # w <= h
                image_curr = rgb_cond.crop((0, int(h_in/2-w_in/2), w_in, int(h_in/2+w_in/2))).resize((self.cam.W, self.cam.H))
        ########## Begining of PCD generation
        render_poses = get_pcdGenPoses(pcdgenpath) # n * 4 * 4的相机位姿
        if depth_cond is not None:
            depth_curr = depth_cond
        else:
            depth_curr = self.d(image_curr) # 当前的deth map
        # deoth_curr = align_depth(deoth_curr, lidar_map)
        center_depth = np.mean(depth_curr[h_in//2-10:h_in//2+10, w_in//2-10:w_in//2+10])

        ###########################################################################################################################
        # Iterative scene generation
        H, W, K = self.cam.H, self.cam.W, self.cam.K

        x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
        edgeN = 2
        edgemask = np.ones((H-2*edgeN, W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        ### initialize : calculate the first point cloud and color by lifting pixels using predicted depth model
        R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
        new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2

        pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

        if self.for_gradio:
            progress(0, desc='[1/4] Dreaming...')
            iterable_dream = progress.tqdm(range(1, len(render_poses)), desc='[1/4] Dreaming')
        else:
            iterable_dream = range(1, len(render_poses))

        logger.info("start to generate pcd") 
        logger.info(f"number of images to generate: {len(iterable_dream)}")  
        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        frames = []
        for i in iterable_dream:
            logger.info(f"generating the {i}th image")
            R, T = render_poses[i,:3,:3], render_poses[i,:3,3:4]
            
            R_for_cam = np.matmul(yz_reverse, R).T
            T_for_cam = -np.matmul(R_for_cam, np.matmul(yz_reverse, T))
            c2w_for_cam = np.concatenate((R_for_cam, T_for_cam), axis=1)
            c2w_for_cam = np.concatenate((c2w_for_cam, np.array([[0,0,0,1]])), axis=0)


            ### Transform world to pixel
            pts_coord_cam2 = R.dot(pts_coord_world) + T  ### Same with c2w*world_coord (in homogeneous space)
            pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)   #.reshape(3,H,W).transpose(1,2,0).astype(np.float32)

            depth_map, valid_idx_map = pointcloud_to_depth_map_and_valid_idx(pts_coord_world.T, rotation_mat=R, trans_vec=T, K=K, height=H, width=W)
            valid_idx_map = valid_idx_map.reshape(-1)
            valid_idx_map = valid_idx_map[valid_idx_map != -1]
            # To true and false
            valid_idx = valid_idx_map
            
            # valid_idx = np.zeros_like(pts_coord_world[0], dtype=bool)
            # valid_idx[valid_idx_map] = True

            depth_map = Image.fromarray((depth_map[..., None].repeat(3, axis=-1) / depth_map.max() * 255.).astype(np.uint8))
            
            # valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
            #                                             pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
            #                                             pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, # originally -1 
            #                                             pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
            #                                             pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0] # originally -1
            # import IPython; IPython.embed(header="maybe optimize here...")
            pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
            round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)
            image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,3)
            image2 = edgemask[...,None]*image2 + (1-edgemask[...,None])*np.pad(image2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            round_mask2 = np.zeros((H,W), dtype=np.float32)
            round_mask2[round_coord_cam2[1], round_coord_cam2[0]] = 1

            round_mask2 = maximum_filter(round_mask2, size=(9,9), axes=(0,1))
            image2 = round_mask2[...,None]*image2 + (1-round_mask2[...,None])*(-1)

            mask2 = minimum_filter((image2.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
            image2 = mask2[...,None]*image2 + (1-mask2[...,None])*0

            mask_hf = np.abs(mask2[:H-1, :W-1] - mask2[1:, :W-1]) + np.abs(mask2[:H-1, :W-1] - mask2[:H-1, 1:])
            mask_hf = np.pad(mask_hf, ((0,1), (0,1)), 'edge')
            mask_hf = np.where(mask_hf < 0.3, 0, 1)
            border_valid_idx = np.where(mask_hf[round_coord_cam2[1], round_coord_cam2[0]] == 1)[0]  # use valid_idx[border_valid_idx] for world1

            diff_input_mask = np.ones_like(mask2)
            diff_input_mask *= (mask2 < 0.5)

            image2 = Image.fromarray((image2 * 255.).astype(np.uint8))
            mask_for_rgb = Image.fromarray((diff_input_mask * 255.).astype(np.uint8))

            mask_blur, image_curr_ori = self.rgb(
                image=image2,
                control_image=depth_map,
                mask_image=mask_for_rgb,
            )
        
            if args.reinpaint:
                image_curr = self.reinpainter(
                    image=image_curr_ori,
                    noise_level=150,
                )
            else:
                image_curr = image_curr_ori
            image_curr_tmp = image_curr.copy()
            depth_curr = self.d(image_curr)
            ##### save rgb to a path
            depth_curr_map = Image.fromarray((depth_curr[..., None].repeat(3, axis=-1) / depth_curr.max() * 255.).astype(np.uint8))
            # depth_curr_map.save(f"new_inference_depth_{i}.png")

            mask_blur, image_curr_ori = self.rgb(
                image=image2,
                control_image=depth_curr_map,
                mask_image=mask_for_rgb,
            )
            if args.reinpaint:
                image_curr = self.reinpainter(
                    image=image_curr_ori,
                    noise_level=150,
                )
            else:
                image_curr = image_curr_ori
            
            image_curr_copy = image_curr.copy()
            depth_curr = self.d(image_curr_copy)
            
            frames.append({'image': image_curr, 'transform_matrix':c2w_for_cam, 'depth': depth_curr})
            
            ##### save rgb to a path
            depth_curr_map = Image.fromarray((depth_curr[..., None].repeat(3, axis=-1) / depth_curr.max() * 255.).astype(np.uint8))
            # depth_curr_map.save(f"new_inference_depth_{i}.png")
            

            if not os.path.exists(args.save_inpainted_images_path):
                os.mkdir(args.save_inpainted_images_path)
            image_curr_tmp.save(os.path.join(args.save_inpainted_images_path, f"{i}_1.png"))
            image_curr.save(os.path.join(args.save_inpainted_images_path, f"{i}_2.png"))
            if not os.path.exists(args.save_warped_depth_path):
                os.mkdir(args.save_warped_depth_path)
            depth_map.save(os.path.join(args.save_warped_depth_path, f"{i}.png"))
            if not os.path.exists(args.save_mask_path):
                os.mkdir(args.save_mask_path)
            mask_for_rgb.save(os.path.join(args.save_mask_path, f"{i}.png"))
            if not os.path.exists(args.save_mask_blur_path):
                os.mkdir(args.save_mask_blur_path)
            mask_blur.save(os.path.join(args.save_mask_blur_path, f"{i}.png"))
            

            ### depth optimize
            t_z2 = torch.tensor(depth_curr)
            sc = torch.ones(1).float().requires_grad_(True)
            trans = torch.zeros(3).float().requires_grad_(True)
            
            optimizer = torch.optim.Adam(params=[sc, trans], lr=0.0005)
            loss_last = 999
            idx = 0
            
            while True:
                # trans3d = torch.tensor([[sc,0,0,0], [0,sc,0,0], [0,0,sc,0], [0,0,0,1]],dtype=torch.float32).requires_grad_(True)
                trans3d = torch.stack([torch.cat([sc.view(1), torch.zeros(2), 0*trans[0].view(1)]), 
                       torch.cat([torch.zeros(1), sc.view(1), torch.zeros(1), trans[1].view(1)]), 
                       torch.cat([torch.zeros(2), sc.view(1), 0*trans[2].view(1)]), 
                       torch.tensor([0, 0, 0, 1], dtype=torch.float32)])

                # trans3d.requires_grad = True
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1], round_coord_cam2[0]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1,valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]
                err = (torch.tensor(pts_coord_world[:,valid_idx]).float() - coord_world2_trans)**2
                err_sorted, idx_sorted = torch.sort(err.sum(0), descending=False)
                loss = torch.mean(err_sorted[:int(err_sorted.shape[0]*0.75)])
                optimizer.zero_grad()
                loss.backward()
                if idx == 0 or idx % 10 == 0:
                    print(f"loss_{idx}: ", loss.item())
                idx += 1
                optimizer.step()
 
                if idx > 100:
                    if torch.abs(loss_last - loss) / loss < 0.0001:
                        break
                if idx > 2000:
                    break
                loss_last = loss.item()
            print("loss_final: ", loss.item())
            
            with torch.no_grad():
                coord_cam2 = torch.matmul(torch.tensor(np.linalg.inv(K)), torch.stack((torch.tensor(x)*t_z2, torch.tensor(y)*t_z2, 1*t_z2), axis=0)[:,round_coord_cam2[1, border_valid_idx], round_coord_cam2[0, border_valid_idx]].reshape(3,-1))
                coord_world2 = (torch.tensor(np.linalg.inv(R)).float().matmul(coord_cam2) - torch.tensor(np.linalg.inv(R)).float().matmul(torch.tensor(T).float()))
                coord_world2_warp = torch.cat((coord_world2, torch.ones((1, border_valid_idx.shape[0]))), dim=0)
                coord_world2_trans = torch.matmul(trans3d, coord_world2_warp)
                coord_world2_trans = coord_world2_trans[:3] / coord_world2_trans[-1]  # border pixels


            trans3d = trans3d.detach().numpy()

            pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]] # View2. camera space
            camera_origin_coord_world2 = - np.linalg.inv(R).dot(T).astype(np.float32) # 3, 1
            new_pts_coord_world2 = (np.linalg.inv(R).dot(pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1] # View2. PCs.
            new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]  # View 2 Colors
            
            # # I want to output this in PCD!
            
            pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
            pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)
            

            # vector_camorigin_to_campixels = coord_world2_trans.detach().numpy() - camera_origin_coord_world2 # 
            # vector_camorigin_to_pcdpixels = pts_coord_world[:,valid_idx[border_valid_idx]] - camera_origin_coord_world2

            # compensate_depth_coeff = np.sum(vector_camorigin_to_pcdpixels * vector_camorigin_to_campixels, axis=0) / np.sum(vector_camorigin_to_campixels * vector_camorigin_to_campixels, axis=0) # N_correspond
            # compensate_pts_coord_world2_correspond = camera_origin_coord_world2 + vector_camorigin_to_campixels * compensate_depth_coeff.reshape(1,-1)

            # compensate_coord_cam2_correspond = R.dot(compensate_pts_coord_world2_correspond) + T
            # homography_coord_cam2_correspond = R.dot(coord_world2_trans.detach().numpy()) + T

            # compensate_depth_correspond = compensate_coord_cam2_correspond[-1] - homography_coord_cam2_correspond[-1] # N_correspond
            # compensate_depth_zero = np.zeros(4)
            # compensate_depth = np.concatenate((compensate_depth_correspond, compensate_depth_zero), axis=0)  # N_correspond+4

            # pixel_cam2_correspond = pixel_coord_cam2[:, border_valid_idx] # 2, N_correspond (xy)
            # pixel_cam2_zero = np.array([[0,0,W-1,W-1],[0,H-1,0,H-1]])
            # pixel_cam2 = np.concatenate((pixel_cam2_correspond, pixel_cam2_zero), axis=1).transpose(1,0) # N+H, 2

            # # Calculate for masked pixels
            # masked_pixels_xy = np.stack(np.where(1-mask2), axis=1)[:, [1,0]]
            # new_depth_linear, new_depth_nearest = interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy), interp_grid(pixel_cam2, compensate_depth, masked_pixels_xy, method='nearest')
            # new_depth = np.where(np.isnan(new_depth_linear), new_depth_nearest, new_depth_linear)

            # pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))[:,np.where(1-mask2.reshape(-1))[0]]
            # x_nonmask, y_nonmask = x.reshape(-1)[np.where(1-mask2.reshape(-1))[0]], y.reshape(-1)[np.where(1-mask2.reshape(-1))[0]]
            # compensate_pts_coord_cam2 = np.matmul(np.linalg.inv(K), np.stack((x_nonmask*new_depth, y_nonmask*new_depth, 1*new_depth), axis=0))
            # new_warp_pts_coord_cam2 = pts_coord_cam2 + compensate_pts_coord_cam2

            # new_pts_coord_world2 = (np.linalg.inv(R).dot(new_warp_pts_coord_cam2) - np.linalg.inv(R).dot(T)).astype(np.float32)
            # new_pts_coord_world2_warp = np.concatenate((new_pts_coord_world2, np.ones((1, new_pts_coord_world2.shape[1]))), axis=0)
            # new_pts_coord_world2 = np.matmul(trans3d, new_pts_coord_world2_warp)
            # new_pts_coord_world2 = new_pts_coord_world2[:3] / new_pts_coord_world2[-1]
            # new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.)[np.where(1-mask2.reshape(-1))[0]]

            # pts_coord_world = np.concatenate((pts_coord_world, new_pts_coord_world2), axis=-1) ### Same with inv(c2w) * cam_coord (in homogeneous space)
            # pts_colors = np.concatenate((pts_colors, new_pts_colors2), axis=0)

        #################################################################################################

        
        
        traindata = {
            'camera_angle_x': self.cam.fov[0],
            'W': W,
            'H': H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            # 'frames': frames,
            'frames': frames
        }
        
        # write_to_pc_txt(pts_coord_world, f"pts_coord_world_{i}.txt")
        # write_to_pc_txt(new_pts_coord_world2, f"new_pts_coord_world2_{i}.txt")
        # import IPython; IPython.embed()
        # write_to_pc_txt(pts_coord_world, f"pts_coord_world.txt", color=pts_colors)


        # render_poses = get_pcdGenPoses(pcdgenpath)
        # internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': center_depth})

        # if self.for_gradio:
        #     progress(0, desc='[2/4] Aligning...')
        #     iterable_align = progress.tqdm(range(len(render_poses)), desc='[2/4] Aligning')
        # else:
        #     iterable_align = range(len(render_poses))


        # for i in iterable_align:
        #     for j in range(len(internel_render_poses)):
        #         idx = i * len(internel_render_poses) + j
        #         print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')

        #         ### Transform world to pixel
        #         Rw2i = render_poses[i,:3,:3]
        #         Tw2i = render_poses[i,:3,3:4]
        #         Ri2j = internel_render_poses[j,:3,:3]
        #         Ti2j = internel_render_poses[j,:3,3:4]

        #         Rw2j = np.matmul(Ri2j, Rw2i)
        #         Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

        #         # Transfrom cam2 to world + change sign of yz axis
        #         Rj2w = np.matmul(yz_reverse, Rw2j).T
        #         Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
        #         Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
        #         Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

        #         pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
        #         pixel_coord_camj = np.matmul(K, pts_coord_camj)

        #         valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
        #                                                     pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
        #                                                     pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1, 
        #                                                     pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
        #                                                     pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
        #         if len(valid_idxj) == 0:
        #             continue
        #         pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
        #         pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
        #         round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)


        #         x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
        #         grid = np.stack((x,y), axis=-1).reshape(-1,2)
        #         imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
        #         imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

        #         depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
        #         depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')

        #         maskj = np.zeros((H,W), dtype=np.float32)
        #         maskj[round_coord_camj[1], round_coord_camj[0]] = 1
        #         maskj = maximum_filter(maskj, size=(9,9), axes=(0,1))
        #         imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

        #         maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11), axes=(0,1))
        #         imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0
                

        #         # TODO: 我不想要这里的image，不知道它们是怎么来的， 我只想要直接从rgb_model生成的image和相应的poses.
        #         traindata['frames'].append({
        #             'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)), 
        #             'transform_matrix': Pc2w.tolist(),
        #         })
                

        # progress(1, desc='[3/4] Baking Gaussians...')
        save_path = args.save_training_images_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for x, img in enumerate(traindata['frames']):
            i = img['image']
            i.save(os.path.join(save_path, str(x)+'.png'))
        return traindata
