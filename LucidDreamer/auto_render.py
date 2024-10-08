import os
import subprocess

# 定义输入文件夹和输出文件夹
input_folder = './outputs'
output_folder = './teaser_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历input_folder下的所有文件夹和子文件夹
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.ply'):
            # 构建 .ply 文件的完整路径
            ply_path = os.path.join(root, file)
            
            # 根据文件名构建保存图片的路径
            base_name = os.path.splitext(file)[0]
            rgb_save_path = os.path.join(output_folder, f'{base_name}_rgb.png')
            depth_save_path = os.path.join(output_folder, f'{base_name}_depth.png')
            
            # 构建运行render_pic.py的命令
            command = [
                'python', 'render_pic.py',
                '--ply_path', ply_path,
                '--rgb_save_path', rgb_save_path,
                '--depth_save_path', depth_save_path
            ]
            
            # 打印命令（可选）
            print(f"Rendering {ply_path} ...")
            
            # 调用subprocess运行命令
            subprocess.run(command)

print("Rendering complete.")
