import torch
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from PIL import Image
import argparse
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

CAMERAS = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
}

def to_gpu(arr):
    return torch.from_numpy(arr).float().cuda()

def get_camera_pose(per_cam_veh_pose, cam2veh):
    opencv_to_waymo = torch.eye(4, dtype=torch.float32, device='cuda')
    opencv_to_waymo[:3, :3] = torch.tensor([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=torch.float32, device='cuda')
    
    result = torch.matmul(per_cam_veh_pose.float(), cam2veh.float())
    result = torch.matmul(result, opencv_to_waymo)
    
    return torch.inverse(result)

def warp_image_and_depth(image, image_pose, other_pose, K_source, K_target, depth):
    W, H = image.size
    R0, T0 = image_pose[:3, :3], image_pose[:3, 3:4]
    x, y = torch.meshgrid(torch.arange(W, dtype=torch.float32, device='cuda'),
                          torch.arange(H, dtype=torch.float32, device='cuda'), indexing='xy')
    
    pts_coord_cam = torch.matmul(torch.inverse(K_source), 
                                 torch.stack((x*depth, y*depth, depth), dim=0).reshape(3,-1))
    pts_coord_world = (torch.inverse(R0).mm(pts_coord_cam) - torch.inverse(R0).mm(T0)).float()

    R, T = other_pose[:3, :3], other_pose[:3, 3:4]
    depth_map = pointcloud_to_depth_map(pts_coord_world.T, rotation_mat=R, trans_vec=T, height=H, width=W, K=K_target)
   
    depth_map_normalized = (depth_map / depth_map.max() * 255.).cpu().numpy()
    depth_map_rgb = np.stack([depth_map_normalized] * 3, axis=-1).astype(np.uint8)
    depth_map_image = Image.fromarray(depth_map_rgb)
    
    return depth_map_image

def pointcloud_to_depth_map(point_cloud, rotation_mat, trans_vec, K, height, width):
    pts_coord_cam = rotation_mat.mm(point_cloud.T).T + trans_vec[:, 0]
    projected_points = torch.matmul(K, pts_coord_cam.T).T

    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    depth_map = torch.zeros((height, width), device='cuda')
    
    u = torch.round(projected_points[:, 0]).long()
    v = torch.round(projected_points[:, 1]).long()
    
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[mask], v[mask]
    depths = pts_coord_cam[mask, 2]
    
    depth_map.index_put_((v, u), depths, accumulate=False)

    return depth_map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', type=str, required=True, help='path to .tfrecord file')
    parser.add_argument('--output_dir', type=str, required=True, help='output path')
    parser.add_argument('--zoe_model_path', type=str, default=None, help='zoe_depth model path. this is useful for bad network condition')
    return parser.parse_args()

class Aligner:
    def __init__(self, pred_depth, lidar, lidar_mask):
        self.pred_depth = pred_depth
        self.lidar = lidar
        self.lidar_mask = lidar_mask
        self.H, self.W = pred_depth.shape
        self.model = None

    def align(self):
        pred_pixels = self.pred_depth[self.lidar_mask].reshape(-1, 1)
        lidar_pixels = self.lidar[self.lidar_mask].reshape(-1, 1)

        self.model = LinearRegression()
        self.model.fit(pred_pixels, lidar_pixels)

        aligned_depth = self.model.predict(self.pred_depth.reshape(-1, 1))
        aligned_depth = aligned_depth.reshape(self.H, self.W)
        return aligned_depth


def main():
    args = parse_args()
    tfrecord_path = args.tfrecord_path
    output_dir = args.output_dir
    
    record_flag = tfrecord_path.split('/')[-1].split('.')[0]

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'warped_depth'), exist_ok=True)

    # Load ZoeDepth model
    if args.zoe_model_path is not None:
        model = torch.hub.load(args.zoe_model_path, "ZoeD_K", source="local", pretrained=True)
    else:
        model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model.to(DEVICE)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    frames = list(dataset)
    total_frames = len(frames)

    start_frame = 0
    end_frame = total_frames - 1

    for i in range(start_frame, end_frame, 5):
        current_frame = frames[i]
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(current_frame.numpy()))

        for camera_image in frame.images:
            if open_dataset.CameraName.Name.Name(camera_image.name) in CAMERAS:
                camera_name = open_dataset.CameraName.Name.Name(camera_image.name)

                im_array = tf.image.decode_jpeg(camera_image.image).numpy()
                image_filename = os.path.join(output_dir, 'images', f'{record_flag}'+f'{i:03d}_{camera_name}.png')
                image = Image.fromarray(im_array)
                image.save(image_filename)

                # Generate lidar depth map
                calib = frame.context.camera_calibrations[camera_image.name - 1]
                image_shape = (calib.height, calib.width)
                
                (range_images, camera_projections, _, range_image_top_pose) = \
                    frame_utils.parse_range_image_and_camera_projection(frame)
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose)
                
                points_all = np.concatenate(points, axis=0)
                cp_points_all = np.concatenate(cp_points, axis=0)
                points_all_distances = np.linalg.norm(points_all, axis=-1, keepdims=True)
                
                mask = cp_points_all[..., 0] == camera_image.name
                camera_cp_points = cp_points_all[mask]
                camera_points_distances = points_all_distances[mask]
                projected_points = np.concatenate([camera_cp_points[..., 1:3], camera_points_distances], axis=-1)
                
                lidar_depth = np.zeros(image_shape, dtype=np.float32)
                for point in projected_points:
                    x, y, depth = point
                    x, y = int(x), int(y)
                    if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                        if lidar_depth[y, x] == 0 or depth < lidar_depth[y, x]:
                            lidar_depth[y, x] = depth

                # Predict depth using ZoeDepth
                predicted_depth = zoe.infer_pil(image)

                # Align predicted depth with lidar depth
                lidar_mask = lidar_depth > 0
                aligner = Aligner(predicted_depth, lidar_depth, lidar_mask)
                aligned_depth = aligner.align()

                # Prepare camera pose and intrinsics
                camera_calibration = next(c for c in frame.context.camera_calibrations if c.name == camera_image.name)
                K_source = to_gpu(np.array([
                    [camera_calibration.intrinsic[0], 0, camera_calibration.intrinsic[2]],
                    [0, camera_calibration.intrinsic[1], camera_calibration.intrinsic[3]],
                    [0, 0, 1]
                ]))
                ex_source = to_gpu(np.reshape(np.array(camera_calibration.extrinsic.transform), [4, 4]))
                veh_pose_source = to_gpu(np.reshape(np.array(camera_image.pose.transform), [4, 4]))
                image_pose = get_camera_pose(veh_pose_source, ex_source)

                # Warp to future frames
                for j in range(1, 6):
                    other_frame_index = min(i + 5 + j, end_frame)
                    if other_frame_index >= len(frames):
                        break

                    other_frame = frames[other_frame_index]
                    other_frame_data = open_dataset.Frame()
                    other_frame_data.ParseFromString(bytearray(other_frame.numpy()))

                    for other_camera_image in other_frame_data.images:
                        if other_camera_image.name == camera_image.name:
                            other_camera_calibration = next(c for c in other_frame_data.context.camera_calibrations if c.name == other_camera_image.name)
                            K_target = to_gpu(np.array([
                                [other_camera_calibration.intrinsic[0], 0, other_camera_calibration.intrinsic[2]],
                                [0, other_camera_calibration.intrinsic[1], other_camera_calibration.intrinsic[3]],
                                [0, 0, 1]
                            ]))
                            ex_target = to_gpu(np.reshape(np.array(other_camera_calibration.extrinsic.transform), [4, 4]))
                            veh_pose_target = to_gpu(np.reshape(np.array(other_camera_image.pose.transform), [4, 4]))
                            other_pose = get_camera_pose(veh_pose_target, ex_target)

                            warped_depth = warp_image_and_depth(image, image_pose, other_pose, K_source, K_target, to_gpu(aligned_depth))

                            output_path = os.path.join(output_dir, 'warped_depth', f'{record_flag}' + f'{other_frame_index:03d}_{camera_name}_warp.png')
                            warped_depth.save(output_path)
                            break

if __name__ == "__main__":
    main()