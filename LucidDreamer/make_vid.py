import os
import glob
import subprocess

def main():
    # 替换为您的图片文件夹路径和文本文件路径
    image_dir = '/data/tandw/waymo_new/segment-3988957004231180266_5566_500_5586_500_with_camera_labels/RGB_image'
    text_file = '/data/xiesr/lucidsim/LucidSimNana/LucidSim/text.txt'

    # 获取所有图片文件的路径，假设图片为 PNG 格式
    image_files = sorted(glob.glob(os.path.join(image_dir, '*FRONT.png')))

    # 遍历图片文件，每隔10张运行一次脚本
    for idx, image_file in enumerate(image_files):
        if idx > 10:
            if idx % 6  == 0:
                cmd = [
                    'python', 'run.py',
                    '--image', image_file,
                    '--text', text_file,
                    '--reinpaint',
                    # '--campath_render','snake_back',
                    '--save_dir', './outputs/vanilla_autodrive'
                ]
                print(f"运行命令: {' '.join(cmd)}")
                subprocess.run(cmd)

if __name__ == '__main__':
    main()