import argparse
import os
import glob
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config

sys.path.append(str(Path.joinpath(Path.cwd(),"models/PatchFusion")))
sys.path.append(str(Path.joinpath(Path.cwd(),"models/PatchFusion/external")))

from models.models import PatchFusion

# Load the saved calibration parameters
calibration_data = np.load("s23camera.npz")

# Extract the camera matrix
camera_matrix = calibration_data['Camera_matrix']

# The camera matrix typically looks like this:
# [[fx,  0, cx],
#  [ 0, fy, cy],
#  [ 0,  0,  1]]

# Extract the focal lengths
FX = camera_matrix[0, 0]  # Focal length in the x direction
FY = camera_matrix[1, 1]  # Focal length in the y direction
FL = (FX + FY) / 2  # Average focal length

# Print the extracted parameters
print(f"FX: {FX}")
print(f"FY: {FY}")
print(f"FL (average focal length): {FL}")

NYU_DATA = False
INPUT_DIR = './tree/input'
OUTPUT_DIR = './tree-patchfusion/output'
#DATASET = 'nyu'  # For INDOOR
DATASET = 'kitti'  # For OUTDOOR


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        # try:
        color_image = Image.open(image_path).convert('RGB')
        original_width, original_height = color_image.size
        FINAL_HEIGHT = original_height
        FINAL_WIDTH = original_width
        #image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(
        #    'cuda' if torch.cuda.is_available() else 'cpu')

        image_tensor = color_image
        pred = model.run_metric_depth_estimation(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.squeeze().detach().cpu().numpy()

        # Resize color image and depth to final size
        resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
        resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

        focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
        x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
        x = (x - FINAL_WIDTH / 2) / focal_length_x
        y = (y - FINAL_HEIGHT / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(
            os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)
        # except Exception as e:
        #     print(f"Error processing {image_path}: {e}")


def main(model_name):
    #config = get_config(model_name, "eval", DATASET)
    #config.pretrained_resource = pretrained_resource
    #model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatchFusion(device='cuda', model_name=model_name)
    #model.eval()
    process_images(model)


if __name__ == '__main__':
    model_name = "zhyever/patchfusion_depth_anything_vitb14"
    #model_name = "zhyever/patchfusion_zoedepth"
    main(model_name)
