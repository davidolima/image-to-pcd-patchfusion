import torch
import numpy as np
import cv2
import torchvision
import typing as T
import sys

from typing import Literal
from pathlib import Path
from PIL.Image import Image
from numpy import ndarray
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

#from segmentation.model import OwlVIT_SAM
from models.model_interface import GenericDepthModel

# HACK: this is necessary to import the modules from the PatchFusion repo
# so stuff does not break with dozens of "X already registered in Y" mmengine errors
# effectively equivalent to modifying the PYTHONPATH environment variable
# (exposes imports from PatchFusion module)
# str conversion is needed because sys.path can only have strings
sys.path.insert(0, str(Path(__file__).parents[2] / "PatchFusion"))
sys.path.insert(0, str(Path(__file__).parents[2] / "PatchFusion" / "external"))

from estimator.models.patchfusion import PatchFusion as BasePatchFusion
        

class PatchFusion(GenericDepthModel):
    def __init__(self, device: str, model_name:str = "zhyever/patchfusion_depth_anything_vitb14"):
        self.device = device
        self.model_name = model_name
        self.model = BasePatchFusion.from_pretrained(self.model_name).to(device)
        self.model.eval()
        self.default_resolution = self.model.tile_cfg['image_raw_shape']
        self.image_resizer = self.model.resizer

        #self.segmentation_model = OwlVIT_SAM(device=device)

    def run_metric_depth_estimation(self, image: Image, **kwargs) -> ndarray:
        # float normalized (0,1) array
        image_tensor = torchvision.transforms.ToTensor()(np.asarray(image) / 255.0)

        image_lr = self.image_resizer(image_tensor.unsqueeze(dim=0)).float().to(self.device)
        image_hr = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(dim=0), self.default_resolution, mode='bicubic', align_corners=True
        ).float().to(self.device)

        mode = 'r128'
        process_num = 4 # batch process size. It could be larger if the GPU memory is larger
        depth_prediction, _ = self.model(mode='infer', cai_mode=mode, process_num=process_num, image_lr=image_lr, image_hr=image_hr)
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction, image_tensor.shape[-2:]
        )[0, 0].detach().cpu().numpy() # depth shape would be (h, w), similar to the input image.
        return depth_prediction
