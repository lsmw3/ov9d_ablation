import os
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from custom_trainer import CustomTrainer

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions

import wandb
import cv2
from tqdm import tqdm


class SaveOnNanCallback(ModelCheckpoint):
    def __init__(self, dirpath="logs/debugs", filename="nan_detected_{step}"):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if outputs contain NaN values
        if outputs is not None:
            # Assuming outputs is a dictionary with 'loss' key
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                # If outputs is just the loss tensor
                loss = outputs
                
            if trainer.global_step >= 300:
                if torch.isnan(loss).any():
                    # NaN detected, save checkpoint
                    checkpoint_path = f"{self.dirpath}/{self.filename.format(step=trainer.global_step)}.ckpt"
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"NaN detected at step {trainer.global_step}. Checkpoint saved to {checkpoint_path}")
                else:
                    checkpoint_path = f"logs/steps_before_nan/{self.filename.format(step=trainer.global_step)}.ckpt"
                    trainer.save_checkpoint(checkpoint_path)


def main():
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

        # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset, 
        'data_path': args.data_path, 
        'data_name': args.data_name, 
        'data_type': args.data_train,
        'feat_3d_path': args.data_3d_feat ,
        'xyz_bin': args.nocs_bin,
        'raw_w': args.raw_w,
        'raw_h': args.raw_h
    }
    dataset_kwargs['scale_size'] = args.scale_size

    train_dataset = get_dataset(**dataset_kwargs)
    
    def draw_3d_bbox_with_axes(image, keypoints_3d, cam_R_m2c, cam_t_m2c, cam_K):
        """Draws a 3D bounding box and orientation axes on the image."""

        # Convert keypoints to 2D (projected using camera intrinsics)
        keypoints_2d = (keypoints_3d @ cam_R_m2c.T + cam_t_m2c) @ cam_K.T
        keypoints_2d = keypoints_2d[:, :2] / keypoints_2d[:, 2:]  # Convert to homogeneous coordinates

        bbox_edges = [
            (1, 2), (2, 4), (4, 3), (3, 1),  # Bottom face
            (5, 6), (6, 8), (8, 7), (7, 5),  # Top face
            (1, 5), (2, 6), (3, 7), (4, 8)   # Vertical edges
        ]

        # Draw edges of the 3D bounding box
        for i, j in bbox_edges:
            pt1 = tuple(keypoints_2d[i].astype(int))
            pt2 = tuple(keypoints_2d[j].astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        # Draw orientation axes (X, Y, Z)
        axis_length = 0.1  # Scale factor for axis visualization
        origin_3d = np.array([0, 0, 0])  # The object's origin in model coordinates
        axes_3d = np.array([
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, axis_length]   # Z-axis
        ])  # Shape (3,3)

        axes_2d = (axes_3d @ cam_R_m2c.T + cam_t_m2c) @ cam_K.T  # Project to 2D
        axes_2d = axes_2d[:, :2] / axes_2d[:, 2:]  # Convert to homogeneous coordinate

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # X, Y, Z axes colors

        for i in range(3):
            pt1 = tuple(keypoints_2d[0].astype(int))
            pt2 = tuple(axes_2d[i].astype(int))
            cv2.line(image, pt1, pt2, colors[i], 3)

        return image
    import matplotlib.pyplot as plt
    for idx, batch in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="Loading Dataset", unit="batch"):
        # Extract relevant data from batch
        raw_image = batch['raw_scene'].transpose(1, 2, 0)  # (H, W, 3)
        bbox_center = batch['bbox_center']  # (x, y)
        bbox_size = batch['bbox_size']  # width/height (scale)
        nocs_image = (batch['nocs'].transpose(1, 2, 0) * 255).astype(np.uint8)  # (H, W, 3)
        rgb_image = (batch['image'].transpose(1, 2, 0) * 255).astype(np.uint8)  # Cropped RGB image
        cam_K = batch['cam']  # Camera intrinsics
        cam_R_m2c = batch['gt_r']  # Rotation matrix (3, 3)
        cam_t_m2c = batch['gt_t']  # Translation vector (3,)
        keypoints_3d = batch['kps_3d_m']  # (9, 3)

        # Compute 2D bounding box for visualization
        x_min, y_min = bbox_center - bbox_size / 2
        x_max, y_max = bbox_center + bbox_size / 2


        raw_image_with_bbox = raw_image.copy()
        cv2.rectangle(raw_image_with_bbox, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)      

    
        # Draw on the cropped RGB image
        rgb_image_with_3d_bbox = draw_3d_bbox_with_axes(raw_image.copy(), keypoints_3d, cam_R_m2c, cam_t_m2c, cam_K)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Raw image with 2D bounding box
        axes[0].imshow(raw_image_with_bbox)
        axes[0].set_title("Raw Image with 2D BBox")

        # 2. Cropped RGB with 3D BBox and Orientation
        axes[1].imshow(rgb_image_with_3d_bbox)
        axes[1].set_title("RGB with 3D BBox & Orientation")

        # 3. Cropped NOCS image
        axes[2].imshow(nocs_image)
        axes[2].set_title("Cropped NOCS Image")

        for ax in axes:
            ax.axis("off")

        plt.savefig(f"data_visualize/{batch['class_name']}.png", bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == '__main__':
    main()
