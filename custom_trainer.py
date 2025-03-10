from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss, L1Loss
from mmcv.runner import build_optimizer
import pytorch_lightning as L
from scipy.spatial.transform import Rotation as R

from models.model_wo_sd import OV9D
import utils.logging as logging
from utils.losses import angular_distance, rot_l2_loss
from utils.pose_postprocessing import pose_from_pred_centroid_z
from utils.utils import draw_3d_bbox_with_coordinate_frame


def normalize_3d_points(points: torch.Tensor):
    assert len(points.shape) == 3
    assert points.shape[-1] == 3

    h, w = points.shape[:2]
    points = points.reshape(-1, 3)
    points_centered = points - points.mean(dim=0, keepdim=True)
    points_normed = points_centered / points_centered.norm(dim=1).max()
    points_normed = (points_normed + 1) / 2

    return points_normed.reshape(h, w, 3)


class CustomTrainer(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = OV9D(args)
        self.nocs_loss = logging.AverageMeter()
        self.criterion = SmoothL1Loss(beta=0.1)

        self.total_val_steps = 0

    
    def _step(self, batch, prex, batch_idx):
        input_RGB = batch['image']
        input_MASK = batch['mask'].to(bool)
        # feat_3d = batch['3d_feat']
        feat_2d_bp = batch['pcl_model'] # (b, 480, 480, 3)
        roi_coord_2d = batch['roi_coord_2d'] # (b, 2, 480, 480)
        bbox_wh = batch['bbox_size'] # (b)
        bbox_center = batch['bbox_center'] # (b, 2)
        ratio = batch['resize_ratio'].squeeze(-1) # (b)
        nocs = batch['nocs'].permute(0, 2, 3, 1) # (b, 480, 480, 3)
        dis_sym = batch['dis_sym']
        con_sym = batch['con_sym']
        gt_r = batch['gt_r'] # (b, 3, 3)
        gt_t = batch['gt_t'] # (b, 3)
        cams = batch['cam'] # (b, 3, 3)
        gt_trans_ratio = batch["gt_trans_ratio"] # (B, 3)

        is_train = prex == 'train'

        translation_ratio = 100
        gt_t = gt_t / translation_ratio

        # normalize 3d points
        b = feat_2d_bp.shape[0]
        feat_2d_bp_normed = []
        for i in range(b):
            feat_2d_bp_normed.append(normalize_3d_points(feat_2d_bp[i]))
        feat_2d_bp_normed = torch.stack(feat_2d_bp_normed, dim=0).to(feat_2d_bp)

        preds = self.model(input_RGB, feat_2d_bp_normed, roi_coord_2d, class_ids=batch['class_id'])
        pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)
        pred_r, pred_t = preds['pred_r'], preds['pred_t']

        pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
            pred_r,
            pred_centroids=pred_t[:, :2],
            pred_z_vals=pred_t[:, 2:3], # must be [B, 1]
            roi_cams=cams,
            roi_centers=bbox_center,
            resize_ratios=ratio,
            roi_whs=bbox_wh,
            eps=1e-8,
            is_allo=True,
            z_type='REL'
        )
        
        pred_nocs_list, gt_nocs_list = [], []
        pred_bg_list, gt_bg_list = [], []
        for b in range(batch['image'].shape[0]):
            curr_pred_nocs = pred_nocs[b]
            curr_gt_nocs = nocs[b]
            curr_mask = input_MASK[b]
            curr_pred_nocs_model = curr_pred_nocs[curr_mask]
            curr_pred_bg = curr_pred_nocs[~curr_mask]
            curr_gt_nocs_model = curr_gt_nocs[curr_mask]
            curr_gt_bg = curr_gt_nocs[~curr_mask]
            curr_pcl_m = curr_gt_nocs_model - 0.5  # nocs to pcl
            # discrete symmetry
            curr_dis_sym = dis_sym[b]
            dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
            curr_dis_sym = curr_dis_sym[dis_sym_flag]
            aug_pcl_m = torch.stack([curr_pcl_m], dim=0) # (1, n, 3)
            for sym in curr_dis_sym:
                rot, t = sym[0:3, 0:3], sym[0:3, 3]
                rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
                aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0) # (m, n ,3)
            # continuous symmetry
            curr_con_sym = con_sym[b]
            con_sym_flag = torch.sum(torch.abs(curr_con_sym), dim=(-1)) != 0
            curr_con_sym = curr_con_sym[con_sym_flag]
            for sym in curr_con_sym:
                axis = sym[:3].cpu().numpy()
                angles = np.deg2rad(np.arange(5, 180, 5))
                rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
                rots = torch.from_numpy(R.from_rotvec(rotvecs).as_matrix()).to(curr_pcl_m)
                rot_pcl_m_list = []
                for rot in rots:
                    rot_pcl_m = aug_pcl_m @ rot.T
                    rot_pcl_m_list.append(rot_pcl_m)
                aug_pcl_m = torch.cat([aug_pcl_m] + rot_pcl_m_list, dim=0)
            curr_gt_nocs_set = aug_pcl_m + 0.5
            with torch.no_grad():
                curr_gt_nocs_set = torch.unbind(curr_gt_nocs_set, dim=0)
                loss = list(map(lambda gt_nocs: self.criterion(curr_pred_nocs_model, gt_nocs), curr_gt_nocs_set))
                min_idx = torch.argmin(torch.tensor(loss))
            curr_gt_nocs_model = curr_gt_nocs_set[min_idx]
            
            pred_nocs_list.append(curr_pred_nocs_model)
            gt_nocs_list.append(curr_gt_nocs_model)
            pred_bg_list.append(curr_pred_bg)
            gt_bg_list.append(curr_gt_bg)

        # nocs loss
        loss_o = 2*self.criterion(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list)) + 0.5*self.criterion(torch.cat(pred_bg_list), torch.cat(gt_bg_list))

        # rotation loss
        loss_rot = angular_distance(pred_ego_rot, gt_r)

        #translation loss
        loss_trans = nn.L1Loss(reduction="mean")(pred_trans, gt_t)

        # bind loss
        pred_bind = torch.bmm(pred_ego_rot.permute(0, 2, 1), pred_trans.view(-1, 3, 1)).view(-1, 3)
        gt_bind = torch.bmm(gt_r.permute(0, 2, 1), gt_t.view(-1, 3, 1)).view(-1, 3)
        loss_bind = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)

        # centroid loss
        loss_centroid = nn.L1Loss(reduction="mean")(pred_t[:, :2], gt_trans_ratio[:, :2])

        # z loss
        loss_z = nn.L1Loss(reduction="mean")(pred_t[:, 2], gt_trans_ratio[:, 2]/translation_ratio)

        # total loss
        # loss_total = 2*loss_o + loss_rot + 0.1*loss_trans + 0.1*loss_bind + loss_centroid + 0.1*loss_z
        loss_total = loss_o

        self.nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))

        loss_dict = {
            'loss': loss_total,
            'nocs loss': loss_o.item(),
            'rotation loss': loss_rot.item(),
            'translation loss': loss_trans.item(),
            'Rt loss': loss_bind.item(),
            'centroid loss': loss_centroid.item(),
            'z loss': loss_z.item()
        }

        # self.log(f'{prex}/loss', loss_o, on_step=True, on_epoch=True, prog_bar=True)
        self.log_step(loss_dict, prex)

        if prex == "train":
            if 0 == self.trainer.global_step % 10 and (self.trainer.local_rank == 0):
                output_vis = self.vis_images(preds, batch, pred_ego_rot, pred_trans*translation_ratio)
                for key, value in output_vis.items():
                    imgs = [np.concatenate([img for img in value],axis=0)]
                    self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
        else:
            if 0 == self.total_val_steps % 10:
                output_vis = self.vis_images(preds, batch, pred_ego_rot, pred_trans*translation_ratio)
                for key, value in output_vis.items():
                    imgs = [np.concatenate([img for img in value],axis=0)]
                    self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)

            self.total_val_steps += 1

        torch.cuda.empty_cache()

        return loss_total
    

    def log_step(self, objectives: Dict[str, torch.Tensor], prex: str):
        for key, value in objectives.items():
            self.log(
                f"{prex}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
    

    def vis_images(self, output, batch, pred_ego_rot, pred_trans):
        outputs = {}
        B, C, H, W = batch['image'].shape

        gt_rgb = batch['image'].permute(0, 2, 3, 1).detach().cpu().numpy()
        gt_nocs = batch['nocs'].permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_nocs = output['pred_nocs'].permute(0, 2, 3, 1).detach().cpu().numpy()

        raw_scene = batch['raw_scene'].permute(0, 2, 3, 1).detach().cpu().numpy() # (b, 480, 640, 3), np.uint8
        kps_3d_m = batch['kps_3d_m'].detach().cpu().numpy() # (B, 9, 3)
        cams = batch['cam'].detach().cpu().numpy()

        gt_r = batch['gt_r'].detach().cpu().numpy()
        gt_t = batch['gt_t'].detach().cpu().numpy()
        pred_r = pred_ego_rot.detach().cpu().numpy()
        pred_t = pred_trans.detach().cpu().numpy()

        gt_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, gt_r, gt_t)
        pred_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, pred_r, pred_t)

        outputs.update({"gt rgb":gt_rgb, "gt nocs":gt_nocs, "pred nocs":pred_nocs, "gt pose":gt_pose, "pred pose":pred_pose})

        return outputs
    

    def vis_pose_on_img(self, image:np.ndarray, kps_m: np.ndarray, cam_K: np.ndarray, m2c_R: np.ndarray, m2c_t: np.ndarray):
        b = image.shape[0]
        vis_imgs = []
        for i in range(b):
            img_with_pose = draw_3d_bbox_with_coordinate_frame(image[i], kps_m[i], m2c_R[i], m2c_t[i], cam_K[i])
            vis_imgs.append(img_with_pose)
        
        return np.array(vis_imgs) # (b, H, W, 3)

    
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train', batch_idx)
    

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val', batch_idx)
    
    
    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, param in self.named_parameters():
            if 'bias' in name or 'LayerNorm' in name or 'GroupNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=self.args.lr,
            betas=(0.9, 0.999),
        )

        return {"optimizer": optimizer}