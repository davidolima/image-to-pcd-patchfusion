import torch
import numpy as np
import glob
import os
import open3d as o3d
import logging
import typing

from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from abc import ABC, abstractmethod
from PIL import Image

#from segmentation.model_interface import GenericSegmentationModel
from depth_to_pcd import depth_map_to_pcd

logging.basicConfig(format="[%(levelname)s] %(message)s [%(pathname)s %(funcName)s %(lineno)d]")
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ICP_THRESHOLD = 0.02
VOXEL_SIZE = 5e-2
## https://www.open3d.org/html/tutorial/Advanced/multiway_registration.html
MAX_CORRESPONDENCE_DISTANCE_COARSE = 15 * VOXEL_SIZE
MAX_CORRESPONDENCE_DISTANCE_FINE = 1.5 * VOXEL_SIZE

class GenericDepthModel(ABC):
    model_name: str
    #segmentation_model: GenericSegmentationModel
    TREE_PROMPT = "a single tree"
    POLE_PROMPT = "a single eletrical pole"

    @abstractmethod
    def run_metric_depth_estimation(self, image: Image.Image, **kwargs) -> np.ndarray:
        """
        Run depth estimation on single image. Returns a depth map (np.ndarray) of the original size

        :param Image image: PIL image on which depth-estimation will be run
        """
        pass

    def plot_depth_map(
        self, depth_map: np.ndarray,
        image_path: str = None,
        output_dir: str = None,
        mask: np.ndarray = None,
    ) -> None:
        """
        Plots the 2D depth map for the given model and image.

        If any of `image_path` or `output_dir` is None, show it to the screen.

        :param depth_map: 2D depth map
        :param image_path: image filepath.
        :param output_dir: directory where to save the image.
        """
        # remove useless singleton dimensions
        image_basename = os.path.basename(image_path).split(".")[0] if image_path else None

        depth_map = depth_map.squeeze()
        assert len(depth_map.shape) == 2, f"Depth map must be 2D, got {depth_map.shape}"
        title = f"Depth map generated for model {self.model_name}"
        if image_basename is not None:
            title += " for image " + image_basename
        plt.title(title)
        plt.imshow(
            depth_map,
            cmap='magma',
            interpolation='nearest',
        )
        if (output_dir is not None) and (image_path is not None):
            plt.savefig(os.path.join(output_dir, image_basename + "_depth.png"))
        else:
            plt.show()
        if mask is not None:
            masked_depth_map = mask * depth_map
            title = f"Masked depth map generated for model {self.model_name}"
            if image_basename is not None:
                title += " for image " + image_basename
            plt.title(title)
            plt.imshow(
                masked_depth_map,
                cmap='magma',
                interpolation='nearest',
            )
            if (output_dir is not None) and (image_path is not None):
                plt.savefig(os.path.join(output_dir, image_basename + "_masked_depth.png"))
            else:
                plt.show()


    def process_image_directory(
        self,
        input_dir: str,
        output_dir: str,
        fx: float, fy: float,
    ) -> None:
        image_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))
        for image_path in tqdm(image_paths, desc=f"Processing Images, mapping to {output_dir}"):
            image_raw = Image.open(image_path).convert('RGB')
            image_basename = os.path.basename(image_path).split(".")[0] if image_path else None
            output_path = os.path.join(output_dir, image_basename + ".ply")

            # see depth map before mapping to 3D
            depth_map = self.run_metric_depth_estimation(image_raw, fx=fx, fy=fy)
            with logging_redirect_tqdm([LOG]):
                try:
                    tensor_tree_mask = self.segmentation_model.segment_object(image_raw, prompt=GenericDepthModel.TREE_PROMPT)
                    tree_mask = tensor_tree_mask.cpu().numpy()
                except:
                    LOG.exception(f"Failed to segment tree in image {image_path}. Falling back to zero mask")
                    tree_mask = np.zeros_like(depth_map)
                try:
                    tensor_pole_mask = self.segmentation_model.segment_object(image_raw, prompt=GenericDepthModel.POLE_PROMPT)
                    pole_mask = tensor_pole_mask.cpu().numpy()
                except:
                    LOG.exception(f"Failed to segment pole in image {image_path}. Falling back to zero mask")
                    pole_mask = np.zeros_like(depth_map)
            seg_mask = np.logical_or(tree_mask, pole_mask)
            self.plot_depth_map(depth_map, image_path=image_path, output_dir=output_dir, mask=seg_mask)

            # pass intrinsic camera parameters, just for the sake of completeness
            pcd = depth_map_to_pcd(
                color_image=image_raw,
                depth_map=depth_map,
                fx=fx, fy=fy,
                mask=seg_mask,
            )
            o3d.io.write_point_cloud(output_path, pcd, print_progress=True)

    @staticmethod
    def pairwise_registration(
        source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
    ) -> typing.Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float, float]:
        """
        Code from https://www.open3d.org/html/tutorial/Advanced/multiway_registration.html

        :param source: source point cloud
        :param target: target point cloud
        :return: transformation matrix, information matrix, RMSE for coarse and fine registration
        """
        # Estimate normals for target point cloud using KDTrees
        # necessary for Point-to-Plane ICP registration, which is said to have faster convergence
        target.estimate_normals()

        point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, MAX_CORRESPONDENCE_DISTANCE_COARSE, np.identity(4),
            point_to_plane,
        )
        coarse_rmse = icp_coarse.inlier_rmse
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, MAX_CORRESPONDENCE_DISTANCE_FINE,
            icp_coarse.transformation,
            point_to_plane,
        )
        fine_rmse = icp_fine.inlier_rmse
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, MAX_CORRESPONDENCE_DISTANCE_FINE,
            icp_fine.transformation
        )
        return transformation_icp, information_icp, coarse_rmse, fine_rmse

    @staticmethod
    def pose_graph(
        pcds: typing.List[o3d.geometry.PointCloud],
    ) -> o3d.pipelines.registration.PoseGraph:
        """
        Code from https://www.open3d.org/html/tutorial/Advanced/multiway_registration.html

        :param pcds: list of point clouds
        :return: PoseGraph object with the graph of affine transformations
        """
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in tqdm(range(n_pcds), desc="Base point cloud"):
            for target_id in tqdm(range(source_id + 1, n_pcds), desc="Target point cloud"):
                transformation_icp, information_icp, coarse_rmse, fine_rmse = GenericDepthModel.pairwise_registration(
                    pcds[source_id], pcds[target_id]
                )
                logging.info(f"Pairwise ICP: source={source_id}, target={target_id} completed")
                logging.info(f"    Coarse registration RMSE: {coarse_rmse}. Fine registration RMSE: {fine_rmse}")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=True))
        return pose_graph

    @staticmethod
    def full_registration(
        pcds: typing.List[o3d.geometry.PointCloud],
    ) -> o3d.geometry.PointCloud:
        """
        Code from https://www.open3d.org/html/tutorial/Advanced/multiway_registration.html

        :returns: final point cloud after registration of all intermediate point clouds
        """
        pose_graph = GenericDepthModel.pose_graph(pcds)
        for pcd_index in range(len(pcds)):
            pose_transform = pose_graph.nodes[pcd_index].pose
            # 4x4 float transformation matrix
            pcds[pcd_index].transform(pose_transform)
        base_pcd = pcds[0]
        for new_pcd in pcds[1:]:
            base_pcd += new_pcd
        return base_pcd

    @staticmethod
    def get_pcd_path_multi_view(view_directory: str, output_dir: str) -> str:
        view_directory = view_directory.rstrip("/")  # remove trailing "/"
        output_path = os.path.join(output_dir, os.path.basename(view_directory) + ".ply")
        return output_path

    def multi_view_reconstruction(
        self,
        view_directory: str,
        output_dir: str,
        fx: float, fy: float,
    ) -> None:
        image_paths = glob.glob(os.path.join(view_directory, '*.png')) + glob.glob(os.path.join(view_directory, '*.jpg'))

        # get base point cloud
        output_path = self.get_pcd_path_multi_view(view_directory, output_dir)
        LOG.info("Processed base point cloud (first view)")
        all_pcds = []

        with logging_redirect_tqdm([LOG]):
            for i, image_path in enumerate(tqdm(
                image_paths,
                desc=f"Processing alternate view of scene '{os.path.basename(view_directory)}', mapping to {output_path}")
            ):
                image_raw = Image.open(image_path).convert('RGB')
                # see depth map before mapping to 3D
                depth_map = self.run_metric_depth_estimation(image_raw, fx=fx, fy=fy)
                tree_mask = self.segmentation_model.segment_object(image_raw, prompt=GenericDepthModel.TREE_PROMPT)
                pole_mask = self.segmentation_model.segment_object(image_raw, prompt=GenericDepthModel.POLE_PROMPT)
                depth_map = depth_map * torch.logical_or(tree_mask, pole_mask).numpy()
                self.plot_depth_map(depth_map, image_path=image_path, output_dir=output_dir)
                pcd = depth_map_to_pcd(
                    color_image=image_raw,
                    depth_map=depth_map,
                    fx=fx, fy=fy,
                )
                # downsample (final merged pcd could be **super** heavy with a lot of views and hundreds of millions of points)
                downsampled_pcd = pcd.voxel_down_sample(VOXEL_SIZE)
                all_pcds.append(downsampled_pcd)
                logging.info(f"Generated point cloud for {i + 1}-th alternate view")
        stitched_icp_pcd = self.full_registration(all_pcds)
        final_downsampled_pcd = stitched_icp_pcd.voxel_down_sample(VOXEL_SIZE)

        # write point cloud
        o3d.io.write_point_cloud(output_path, final_downsampled_pcd, print_progress=True)
