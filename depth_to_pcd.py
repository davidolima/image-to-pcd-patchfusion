import open3d as o3d
import numpy as np

from PIL import Image

def depth_map_to_pcd(
    color_image: Image.Image,
    depth_map: np.ndarray, fx: float, fy: float,
    mask: np.ndarray = None,
) -> o3d.geometry.PointCloud:
    W, H = color_image.size[0], color_image.size[1]
    depth_map = depth_map.squeeze()  # remove channel dimension
    # this is already float image so this conversion is lossless
    depth_image = o3d.geometry.Image(depth_map)

    # FIXME this might screw up scaling, verify this!
    # TODO use extrinsic matrix from camera
    full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_image,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W/2, H/2),
    )
    full_3d_points = np.asarray(full_pcd.points)
    full_colors = np.array(color_image).reshape(-1, 3) / 255.0

    # np.nonzero returns tuple of arrays:
    # first is the indices of the nonzero elements in dimension 0;
    # second is the indices of the nonzero elements in dimension 1;
    # Thus flattened index (since arrays are row-major, C-style) will be 
    # x * depth_map.shape[1] + y
    masked_x, masked_y = np.nonzero(mask * depth_map)
    flattened_masked_indices = masked_x * depth_map.shape[1] + masked_y

    masked_points = full_3d_points[flattened_masked_indices]
    masked_colors = full_colors[flattened_masked_indices]

    # add colors in point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_points)
    pcd.colors = o3d.utility.Vector3dVector(masked_colors)
    # return raw point cloud without downsampling
    return pcd
