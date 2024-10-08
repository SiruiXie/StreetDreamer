import os
import argparse
import time
from PIL import Image
import numpy as np
import cv2
from skimage.transform import resize
from luciddreamer import LucidDreamer
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LucidDreamer')
    # Input options
    parser.add_argument('--image', '-img', type=str, default='examples/Image015_animelakehouse.jpg', help='Input image for scene generation')
    parser.add_argument('--depth', '-d', type=str, default=None, help='Input depth map for scene generation')
    parser.add_argument('--text', '-t', type=str, default='', help='Text prompt for scene generation')
    parser.add_argument('--neg_text', '-nt', type=str, default='', help='Negative text prompt for scene generation')

    # Camera options
    parser.add_argument('--campath_gen', '-cg', type=str, default='autodrive', choices=['lookdown', 'lookaround', 'rotate360', 'moveback', 'forward_facing', 'autodrive'], help='Camera extrinsic trajectories for scene generation')
    parser.add_argument('--campath_render', '-cr', type=str, default='autodrive_render', choices=['back_and_forth', 'llff', 'headbanging', 'moveback', '360', 'forward_facing', 'autodrive_render', 'snake_back'], help='Camera extrinsic trajectories for video rendering')

    # Inpainting options
    parser.add_argument('--model_name', type=str, default=None, help='Model name for inpainting(dreaming)')
    parser.add_argument('--seed', type=int, default=1, help='Manual seed for running Stable Diffusion inpainting')
    parser.add_argument('--diff_steps', type=int, default=50, help='Number of inference steps for running Stable Diffusion inpainting')

    # Save options
    parser.add_argument('--save_dir', '-s', type=str, default='', help='Save directory')
    parser.add_argument('--use_controlnet_inpainter', type=bool, default=True, help='if to use finetuned wapred depth controlnet inpainter')
    parser.add_argument('--save_inpainted_images_path', type=str, default='./inpainted_images', help='Save directory for inpainted images')
    parser.add_argument('--save_training_images_path', type=str, default='./training_images', help='Save directory for training images')
    parser.add_argument('--save_warped_depth_path', type=str, default='./warped_depth', help='Save directory for warped depth maps')
    parser.add_argument('--save_mask_path', type=str, default='./mask_images', help='Save directory for mask images')
    parser.add_argument('--save_mask_blur_path', type=str, default='./mask_images_blur', help='Save directory for mask images with blur')
    parser.add_argument('--reinpaint', action='store_true', help='if to use reinpaint')
    
    parser.add_argument('--tuned_controlnet', action='store_true', help='if to use reinpaint')
    parser.add_argument('--personalized', action='store_true', help='if to use reinpaint')
    parser.add_argument('--ip_adapter', action='store_true', help='if to use reinpaint')
    parser.add_argument('--use_naive_sd', action='store_true', help='if to use naive sd')

    args = parser.parse_args()


    ### input (example)
    rgb_cond = Image.open(args.image)
    if args.depth is not None:
        depth_curr = np.load(args.depth)
        
        H0, W0 = depth_curr.shape
        scale_factor = 512 / min(H0, W0)
        resized = resize(depth_curr, (int(H0 * scale_factor), int(W0 * scale_factor)))
        cropped = resized[(resized.shape[0] - 512) // 2 : (resized.shape[0] + 512) // 2, 
                    (resized.shape[1] - 512) // 2 : (resized.shape[1] + 512) // 2]
        depth_cond = cropped
    else:
        depth_cond = None
    
    if args.text.endswith('.txt'):
        with open(args.text, 'r') as f:
            txt_cond = f.readline()
    else:
        txt_cond = args.text

    if args.neg_text.endswith('.txt'):
        with open(args.neg_text, 'r') as f:
            neg_txt_cond = f.readline()
    else:
        neg_txt_cond = args.neg_text


    img_name = os.path.splitext(os.path.basename(args.image))[0]
    # Make default save directory if blank
    if args.save_dir == '':
        
        args.save_dir = f'./outputs/{img_name}_{args.campath_gen}_{args.seed}'
    if not os.path.exists(args.save_dir):
        args.save_dir = os.path.join(args.save_dir, f'{img_name}_{args.campath_gen}_{args.seed}')
        os.makedirs(args.save_dir, exist_ok=True)

    if args.model_name is not None and args.model_name.endswith('safetensors'):
        print('Your model is saved in safetensor form. Converting to HF models...')
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=args.model_name,
            from_safetensors=True,
            device='cuda',
            )
        pipe.save_pretrained('stablediffusion/', safe_serialization=False)
        args.model_name = f'stablediffusion/{args.model_name}'

    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.save_dir, curr_time)

    ld = LucidDreamer(for_gradio=False, save_dir=save_dir, use_tuned_inpainter=args.use_controlnet_inpainter, args=args)
    ld.create(rgb_cond, depth_cond, txt_cond, neg_txt_cond, args.campath_gen, args.seed, args.diff_steps, model_name=args.model_name, args=args)
    ld.render_video(args.campath_render)
