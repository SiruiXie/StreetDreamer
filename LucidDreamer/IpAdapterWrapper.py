import torch
import numpy as np
from PIL import Image

from diffusers import (
    AutoPipelineForImage2Image, StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline
)

class IpAdapterWrapper:
    def __init__(self, control_model_path, sd_model_path, device='cuda'):
        controlnet = ControlNetModel.from_pretrained(control_model_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(sd_model_path,
                                                                             controlnet=controlnet, torch_dtype=torch.float16).to(device)
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder='models', weight_name='ip-adapter_sd15.bin')
        self.generator = torch.Generator(device=device)
    
    def __call__(self, depth_img_or_url, adaper_image_or_url, mask_pil, prompt, negative_prompt=None):
        if isinstance(depth_img_or_url, str):
            depth_img_or_url = Image.open(depth_img_or_url)
        elif isinstance(depth_img_or_url, np.ndarray):
            depth_img_or_url = Image.fromarray(depth_img_or_url.astype(np.uint8))
        elif isinstance(depth_img_or_url, torch.Tensor):
            depth_img_or_url = Image.fromarray((depth_img_or_url.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
        elif isinstance(depth_img_or_url, Image.Image):
            depth_img_or_url = depth_img_or_url
        else:
            raise ValueError("image_or_url must be a string, numpy array, or torch tensor")
        
        
        if isinstance(adaper_image_or_url, str):
            adaper_image_or_url = Image.open(adaper_image_or_url)
        elif isinstance(adaper_image_or_url, np.ndarray):
            adaper_image_or_url = Image.fromarray(adaper_image_or_url.astype(np.uint8))
        elif isinstance(adaper_image_or_url, torch.Tensor):
            adaper_image_or_url = Image.fromarray((adaper_image_or_url.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
        elif isinstance(adaper_image_or_url, Image.Image):
            adaper_image_or_url = adaper_image_or_url
        else:
            raise ValueError("image_or_url must be a string, numpy array, or torch tensor")
        
        #import pdb; pdb.set_trace()
        images = self.pipe(
            prompt=prompt,
            image=adaper_image_or_url,
            mask_image=mask_pil,
            control_image=depth_img_or_url,
            ip_adapter_image=adaper_image_or_url,
            negative_prompt='monochromes, lowres, bad anatomy, worst quality, low quality',
            generator=self.generator,
            num_inference_steps=50,
        ).images[0]
        
        return mask_pil, images
        
        
