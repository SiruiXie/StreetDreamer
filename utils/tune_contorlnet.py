import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from glob import glob
import time
from typing import List

import warnings
warnings.filterwarnings('ignore')
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint
import transformers
#  import tensorflow as tf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

#TODO####################################### helper functions ########################################

def random_mask(image):
    _, height, width = image.shape
    mask = torch.ones((height, width))
    mask_type = random.choice(['stripes', 'border', 'side', 'random_points'])

    if mask_type == 'stripes':
        max_stripe_width = int(width * 0.2)

        orientation = random.choice(['horizontal', 'vertical'])

        if orientation == 'horizontal':
            stripe_height = random.randint(1, max_stripe_width)
            stripe_y = random.randint(0, height - stripe_height)
            mask[stripe_y:stripe_y + stripe_height, :] = 0
        else:
            stripe_width = random.randint(1, max_stripe_width)
            stripe_x = random.randint(0, width - stripe_width)
            mask[:, stripe_x:stripe_x + stripe_width] = 0

    elif mask_type == 'border':
        max_border_width = int(width * 0.2)

        top = random.randint(0, max_border_width)
        bottom = random.randint(0, max_border_width)
        left = random.randint(0, max_border_width)
        right = random.randint(0, max_border_width)

        mask[:top, :] = 0
        mask[-bottom:, :] = 0
        mask[:, :left] = 0
        mask[:, -right:] = 0

    elif mask_type == 'side':
        max_cover_width = int(width * 0.5)
        cover_width = random.randint(1, max_cover_width)
        side = random.choice(['left', 'right'])
        if side == 'left':
            mask[:, :cover_width] = 0
        else:
            mask[:, -cover_width:] = 0

    elif mask_type == 'random_points':
        density = random.uniform(0.01, 0.2)
        mask = torch.from_numpy(np.random.choice([0, 1], size=(height, width), p=[density, 1 - density])).float()

    masked_image = image * mask.unsqueeze(0)

    return masked_image


def split_dataset_evenly(lists: List[List], train_ratio: float = 0.3):

    dataset_size = len(lists[0])

    train_size = int(dataset_size * train_ratio)

    train_indices = []
    test_indices = []

    step = int(1 / train_ratio)

    for i in range(dataset_size):
        if len(train_indices) < train_size and i % step == 0:
            train_indices.append(i)
        else:
            test_indices.append(i)

    re_l = []
    for l in lists:
        train_list = [l[i] for i in train_indices]
        test_list = [l[i] for i in test_indices]
        re_l.append([train_list, test_list])

    return re_l


def synchronize_lists(full_list, missing_list):

    full_filenames = [img.split('/')[-1].replace('_aligned', '') for img in full_list]
    missing_filenames = [img.split('/')[-1].replace('_aligned', '') for img in missing_list]
    

    full_set = set(full_filenames)
    missing_set = set(missing_filenames)
    missing_images = full_set - missing_set

    synchronized_full_list = [img for img in full_list if img.split('/')[-1] not in missing_images]
    
    return synchronized_full_list, missing_images


class Random_Patched_WaymoDataset(Dataset):
    def __init__(self, args, tokenizer, split='train'):
        self.data_dir = args.dataset_name
        
        self.gt_img_paths = sorted(glob(os.path.join(self.data_dir, 'image', '*.png')))
        self.cond_img_paths = sorted(glob(os.path.join(self.data_dir, 'aligned_depth', '*.png')))
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_cond = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        self.tokenizer = tokenizer
        
        def custom_sort_key(path):
            filename = path.split('/')[-1].replace('_aligned', '')
            return filename
        
        def synchronize_sorted_lists(list1, list2):
            sorted_list1 = sorted(list1, key=custom_sort_key)
            sorted_list2 = sorted(list2, key=custom_sort_key)
            return sorted_list1, sorted_list2
        
        self.gt_img_paths, missing_paths = synchronize_lists(self.gt_img_paths, self.cond_img_paths)
        assert len(self.gt_img_paths) == len(self.cond_img_paths), 'length wrong.'
        self.gt_img_paths, self.cond_img_paths = synchronize_sorted_lists(self.gt_img_paths, self.cond_img_paths)
        if split == 'train':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[:-1000], self.cond_img_paths[:-1000]
        elif split == 'test':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[-1000:], self.cond_img_paths[-1000:]
    
    def __len__(self):
        return len(self.gt_img_paths)
    
    def __getitem__(self, index):
        gt_img_path = self.gt_img_paths[index]
        cond_img_path = self.cond_img_paths[index]

        rgb = self.transform_rgb(np.array(Image.open(gt_img_path)).astype(np.uint8))
        cond = self.transform_cond(np.array(Image.open(cond_img_path)).astype(np.uint8)).repeat(3, 1, 1)
        cond = random_mask(cond)
        input_ids = self.tokenizer(
            text='',
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'input_ids': input_ids} #  we set clip embedding as null string, following most of these works --Sirui
    


class waymoDataset(Dataset):
    def __init__(self, args, tokenizer, split='train'):
        self.data_dir = args.dataset_name
        
        self.gt_img_paths = sorted(glob(os.path.join(self.data_dir, 'image', '*.png')))
        self.cond_img_paths = sorted(glob(os.path.join(self.data_dir, 'aligned_depth', '*.png')))
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_cond = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        self.tokenizer = tokenizer
        
        # there are some problems in dataset.
        def custom_sort_key(path):
            filename = path.split('/')[-1].replace('_aligned', '')
            return filename
        
        def synchronize_sorted_lists(list1, list2):
            sorted_list1 = sorted(list1, key=custom_sort_key)
            sorted_list2 = sorted(list2, key=custom_sort_key)
            return sorted_list1, sorted_list2
        
        assert len(self.gt_img_paths) == len(self.cond_img_paths), 'length wrong.'
        self.gt_img_paths, self.cond_img_paths = synchronize_sorted_lists(self.gt_img_paths, self.cond_img_paths)
        if split == 'train':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[:-1000], self.cond_img_paths[:-1000]
        elif split == 'test':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[-1000:], self.cond_img_paths[-1000:]
    
    def __len__(self):
        return len(self.gt_img_paths)
    
    def __getitem__(self, index):
        gt_img_path = self.gt_img_paths[index]
        cond_img_path = self.cond_img_paths[index]

        rgb = self.transform_rgb(np.array(Image.open(gt_img_path)).astype(np.uint8))
        cond = self.transform_cond(np.array(Image.open(cond_img_path)).astype(np.uint8)).repeat(3, 1, 1)

        
        input_ids = self.tokenizer(
            text='',
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'input_ids': input_ids}
        
class ApoloDataset(Dataset):

    def __init__(self, args, tokenizer):
        self.data_dir = args.dataset_name
        self.gt_img_paths = sorted(glob(os.path.join(self.data_dir, 'stereo_train_*', 'camera_5', '*')))
        self.cond_img_paths = sorted(glob(os.path.join(self.data_dir, 'stereo_train_*', 'disparity', '*')))
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_cond = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        self.tokenizer = tokenizer
        assert len(self.gt_img_paths) == len(self.cond_img_paths), 'length wrong.'

    def __len__(self):
        return len(self.gt_img_paths)

    def __getitem__(self, index):
        gt_img_path = self.gt_img_paths[index]
        cond_img_path = self.cond_img_paths[index]

        rgb = self.transform_rgb(np.array(Image.open(gt_img_path)).astype(np.uint8))
        cond = self.transform_cond(np.array(Image.open(cond_img_path)).astype(np.uint8)).repeat(3, 1, 1)
        
        input_ids = self.tokenizer(
            text='',
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'input_ids': input_ids}

def center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:

    width, height = image.size

    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

#TODO####################################### helper functions and datasets ########################################

class warped_waymoDataset(Dataset):
    def __init__(self, args, tokenizer, split='train', parse_mask=False):
        self.data_dir = args.dataset_name
        self.parse_mask = parse_mask
        self.gt_img_paths = sorted(glob(os.path.join(self.data_dir, 'image_dataset', '*.png')))
        self.cond_img_paths = sorted(glob(os.path.join(self.data_dir, 'warped', '*.png')))
        print(f"Length of gt_img_paths: {len(self.gt_img_paths)}")
        print(f"Length of cond_img_paths: {len(self.cond_img_paths)}")
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_cond = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        self.tokenizer = tokenizer

        def custom_sort_key(path):
            filename = path.split('/')[-1].replace('_warp', '')
            return filename
        
        def synchronize_sorted_lists(list1, list2):
            sorted_list1 = sorted(list1, key=custom_sort_key)
            sorted_list2 = sorted(list2, key=custom_sort_key)
            return sorted_list1, sorted_list2
        
        assert len(self.gt_img_paths) == len(self.cond_img_paths), f'length wrong. gt_img_paths: {len(self.gt_img_paths)}, cond_img_paths: {len(self.cond_img_paths)}'
        self.gt_img_paths, self.cond_img_paths = synchronize_sorted_lists(self.gt_img_paths, self.cond_img_paths)
        
        
        if split == 'train':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[:-1000], self.cond_img_paths[:-1000]
        elif split == 'test':
            self.gt_img_paths, self.cond_img_paths = self.gt_img_paths[-1000:], self.cond_img_paths[-1000:]
    
    def __len__(self):
        return len(self.gt_img_paths)
    
    def __getitem__(self, index):
        gt_img_path = self.gt_img_paths[index]
        cond_img_path = self.cond_img_paths[index]

        rgb = self.transform_rgb(np.array(Image.open(gt_img_path)).astype(np.uint8))
        cond_img = self.transform_cond(np.array(Image.open(cond_img_path)).astype(np.uint8))
        mask = (cond_img[0] == 0).to(torch.float32).unsqueeze(0)
        if self.parse_mask:
            cond = torch.cat([cond_img, mask], dim=0)
        else:
            cond = cond_img
        
        input_ids = self.tokenizer(
            text='',
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'mask': mask,'input_ids': input_ids}
    

class warped_waymoDataset_personalization_traning(Dataset):
    def __init__(self, args, tokenizer, split='train', parse_mask=False, train_data_len=None):
        data_dir = args.dataset_name
        scene_name = args.dataset_name # TODO: modify args help strings. --Sirui
        self.data_dir = os.path.join(data_dir, 'waymo_new', scene_name)
        self.parse_mask = parse_mask
        self.gt_img_paths = sorted(glob(os.path.join(self.data_dir, 'RGB_image', '*.png')))
        self.cond_img_paths = sorted(glob(os.path.join(self.data_dir, 'warped_depth', '*.png')))
        self.gt_depth_paths = sorted(glob(os.path.join(self.data_dir, 'depth', "*.npy")))
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_cond = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        self.transform_gt_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
        ])
        
        self.tokenizer = tokenizer
        
        # there are some problems in dataset.
        def custom_sort_key(path):
            filename = path.split('/')[-1].replace('_warp', '')
            return filename
        
        def synchronize_sorted_lists(list1, list2):
            sorted_list1 = sorted(list1, key=custom_sort_key)
            sorted_list2 = sorted(list2, key=custom_sort_key)
            return sorted_list1, sorted_list2
        
        assert len(self.gt_img_paths) == len(self.cond_img_paths) == len(self.gt_depth_paths), f'length wrong. gt_img_paths: {len(self.gt_img_paths)}, cond_img_paths: {len(self.cond_img_paths)}, gt_depths: {len(self.gt_depth_pathst)}'
        self.gt_img_paths, self.cond_img_paths = synchronize_sorted_lists(self.gt_img_paths, self.cond_img_paths)
        [train_gt_img_paths, test_gt_img_paths], [train_cond_img_paths, test_cond_img_paths], [train_gt_depths, test_gt_depths] = split_dataset_evenly([self.gt_img_paths, self.cond_img_paths, self.gt_depth_paths], train_ratio=0.3)
        
        self.split = split
        if split == 'train':
            if train_data_len is not None:
                assert train_data_len <= len(train_gt_img_paths)
                self.gt_img_paths, self.cond_img_paths, self.gt_depth_paths = train_gt_img_paths[:train_data_len], train_cond_img_paths[:train_data_len], train_gt_depths[:train_data_len]
            else:
                self.gt_img_paths, self.cond_img_paths, self.gt_depth_paths = train_gt_img_paths, train_cond_img_paths, train_gt_depths
        elif split == 'test':
            self.gt_img_paths, self.cond_img_paths, self.gt_depth_paths = test_gt_img_paths, test_cond_img_paths, test_gt_depths
    
    def __len__(self):
        return len(self.gt_img_paths)
    
    def __getitem__(self, index):
        gt_img_path = self.gt_img_paths[index]
        cond_img_path = self.cond_img_paths[index]
        gt_depth = self.gt_depth_paths[index]

        rgb = self.transform_rgb(np.array(Image.open(gt_img_path)).astype(np.uint8))
        cond_img = self.transform_cond(np.array(Image.open(cond_img_path)).astype(np.uint8))
        if self.split == 'test':
            gt_depth = self.transform_gt_depth(np.load(gt_depth, allow_pickle=True).astype(np.uint8))
        
        mask = (cond_img[0] == 0).to(torch.float32).unsqueeze(0)
        if self.parse_mask:
            cond = torch.cat([cond_img, mask], dim=0)
        else:
            cond = cond_img
        
        input_ids = self.tokenizer(
            text='',
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        if self.split == 'train':
            return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'mask': mask,'input_ids': input_ids}
        elif self.split == 'test':
            return {'pixel_values': rgb, 'conditioning_pixel_values': cond, 'mask': mask,'input_ids': input_ids, 'gt_depth': gt_depth}
        else:
            raise ValueError


#TODO####################################### validation ########################################

def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, dataset, is_final_validation=False,
):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet.module,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    
    validation_index = torch.randint(0, len(dataset), (args.num_validation_images,))
    validation_prompts = [''] * args.num_validation_images
    validation_images = []
    validation_gt = []
    for index in validation_index:
        cond_img = dataset[index]['conditioning_pixel_values']
        gt_img = dataset[index]['pixel_values']
        
        gt_img = (gt_img + 1) * 127.5
        gt_img = Image.fromarray(gt_img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        validation_gt.append(gt_img)
        validation_images.append(cond_img.unsqueeze(0))

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    
    for validation_prompt, validation_image, gt in zip(validation_prompts, validation_images, validation_gt):
        with inference_ctx:
            image = pipeline(
                validation_prompt, validation_image, num_inference_steps=20, generator=generator
            ).images[0]
        image_logs.append(
            {"validation_image": validation_image, "image": image, 'gt': gt, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["image"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                gt = log['gt']

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
                formatted_images.append(wandb.Image(images, caption="controlnet predicted image"))
                formatted_images.append(wandb.Image(gt, caption="ground truth"))

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs

#TODO####################################### validation ########################################

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


#TODO####################################### training args ########################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="lllyasviel/sd-controlnet-depth",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        help='previously tuned unet checkpoint path for personaliation'
    )
    parser.add_argument(
        '--controlnet_path',
        type=str,
        help='previously tuned controlnet checkpoint path for personalization'
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fine-tuned-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=200, help='number of max epochs for training. if --max_train_provided, this arguments will be override.')
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=25,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=25,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument( # TODO: remove this arg
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", default=False, help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help='path to the dataset',
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1200,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--parameters_to_optimize",
        type=str,
        default="controlnet+unet_decoder",
        choices=['controlnet', 'controlnet+unet_decoder', 'controlnet+unet'],
        help="The parameters to optimize. Choose between 'unet' and 'controlnet'.",
    )
    parser.add_argument(
        "--cat_mask_to_depth_condition",
        action="store_true",
        help='if to concat 01 mask to depth conditioning while training and infer'
    )
    parser.add_argument(
        "--apply_mask_to_latent",
        action="store_true",
        help='if to concat 01 mask to depth conditioning while training and infer'
    )
    parser.add_argument(
        "--train_data_len",
        type=int,
        default=None,
        help='length of the training data used for personalization'
    )
    parser.add_argument(
        '--training_mode',
        type=str,
        default='baseline',
        choices=['baseline', 'personalization'],
        help='train in baesline tune or personalization'
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.train_data_len is not None and args.training_mode != "personalization":
        raise ValueError(f"train_data_len flag should be None if training mode is not personalization!")


    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def collate_fn(examples):
    
    pixel_values = []
    conditioning_pixel_values = []
    input_ids = []
    mask = []
    gt_depth = []
    for example in examples:
        pixel_values.append(example["pixel_values"].to('cuda'))
        conditioning_pixel_values.append(example["conditioning_pixel_values"].to('cuda'))
        input_ids.append(example["input_ids"].to('cuda'))
        mask.append(example['mask'].to('cuda'))
        if 'gt_depth' in example.keys():
            gt_depth.append(example['gt_depth']).to('cuda')
    
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    conditioning_pixel_values = torch.stack(conditioning_pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack(input_ids).to('cuda')
    mask = torch.stack(mask).to('cuda').to(memory_format=torch.contiguous_format).float()
    gt_depth = torch.stack(gt_depth).to('cuda').to(memory_format=torch.contiguous_format).float() if len(gt_depth) > 0 else None
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "mask": mask,
        "gt_depth": gt_depth
    }
    

#TODO####################################### main function ########################################

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    if args.training_mode =='personalization':
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_path, revision=args.revision, variant=args.variant
        )
        controlnet = ControlNetModel.from_pretrained(args.controlnet_path, revision=args.revision, variant=args.variant)
    elif args.training_mode == 'baseline':
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)

    if args.cat_mask_to_depth_condition:
        conv_in_weight = controlnet.controlnet_cond_embedding.conv_in.weight.data
        new_weights_slice = conv_in_weight[:, [0], :, :].clone()
        
        new_weights = torch.cat([conv_in_weight, new_weights_slice], dim=1)
        new_bias = controlnet.controlnet_cond_embedding.conv_in.bias.data.clone()

        new_controlnet_conv_in = torch.nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        new_controlnet_conv_in.weight = torch.nn.Parameter(new_weights)
        new_controlnet_conv_in.bias = torch.nn.Parameter(new_bias)
        
        controlnet.controlnet_cond_embedding.conv_in = new_controlnet_conv_in
    
    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    # unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    pams = None
    # Optimizer creation
#TODO####################################### trainable functions ########################################
    if args.parameters_to_optimize == 'controlnet':
        params_to_optimize = list(unet.parameters())
        pams = 'controlnet'
    elif args.parameters_to_optimize == 'controlnet+unet_decoder':
        params_to_optimize = list(controlnet.parameters()) + list(unet.up_blocks.parameters()) + list(unet.conv_out.parameters())
        pams = 'controlnet+unet_decoder'
    elif args.parameters_to_optimize == 'controlnet+unet':
        params_to_optimize = list(controlnet.parameters()) + list(unet.parameters())
        pams = 'controlnet+unet'
    else:
        raise ValueError(f"args if 'params_to_optimize' : {args.parameters_to_optimize} not valid")
    # params_to_optimize = list(controlnet.parameters()) + list(unet.up_blocks.parameters()) + list(unet.conv_out.parameters())
    print("paremeters_to_optimize", pams)
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
#TODO####################################### trainable functions ########################################
    if args.training_mode == 'baseline':
        train_dataset = warped_waymoDataset(args, tokenizer,split='train', parse_mask=args.cat_mask_to_depth_condition)
        test_dataset = warped_waymoDataset(args, tokenizer, split='test', parse_mask=args.cat_mask_to_depth_condition)
    elif args.training_mode == 'personalization':
        train_dataset = warped_waymoDataset_personalization_traning(args, tokenizer, split='train', parse_mask=args.cat_mask_to_depth_condition, train_data_len=args.train_data_len)
        test_dataset = warped_waymoDataset_personalization_traning(args, tokenizer, split='test', parse_mask=args.cat_mask_to_depth_condition, train_data_len=args.train_data_len)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    controlnet, unet, optimizer, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, lr_scheduler
    )
    

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    time_log_file_path = os.path.join(args.output_dir, 'step_time_log.txt')
    if not os.path.exists(time_log_file_path):
        os.system(f"touch {time_log_file_path}")
    global_training_time=0
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                step_start_time = time.time()
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                if args.apply_mask_to_latent == True:
                    mask = (batch['mask'] < 0.5).to(torch.float32).to(mid_block_res_sample.device)
                    mask_for_mid_latent = torch.nn.functional.interpolate(
                        mask,
                        size=mid_block_res_sample.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                    mid_block_res_sample = mid_block_res_sample * mask_for_mid_latent
                    
                    new_down_block_res_samples = []
                    for sample in down_block_res_samples:
                        mask_multiplyer = torch.nn.functional.interpolate(
                            mask,
                            size=sample.shape[-2:],
                            mode='bilinear',
                            align_corners=True
                        )
                        new_down_block_res_samples.append(sample * mask_multiplyer)
                else:
                    new_down_block_res_samples = down_block_res_samples


                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in new_down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            step_time = time.time()
            global_training_time += (step_time - step_start_time)
            # with open(time_log_file_path, 'w') as time_log_file:
            #     time_log_file.write(f"step {step} costs time {(step_time - step_start_time):.2f}, global time cost (including evaluation and cloud pushing) {global_training_time:.2f}")

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        unet_save_path = os.path.join(args.output_dir, f"checkpoint-unet-{global_step+175}")
                        controlnet_save_paht = os.path.join(args.output_dir, f"checkpoint-controlnet-{global_step+175}")
                        unet = unet.eval()
                        controlnet = controlnet.eval()

                        controlnet_to_save = controlnet.module if hasattr(controlnet, 'module') else controlnet
                        unet_to_save = unet.module if hasattr(unet, 'module') else unet
                        
                        controlnet_to_save.save_pretrained(controlnet_save_paht)
                        unet_to_save.save_pretrained(unet_save_path)
 
                        unet = unet.train()
                        controlnet = controlnet.train()
                        # accelerator.save_state(save_path)
                        logger.info(f"unet saved state to {unet_save_path}, controlnet saved state to {controlnet_save_paht}")

                    if args.validation_prompt is not None and (global_step % args.validation_steps == 0 or global_step == 1):
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            dataset=test_dataset
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)