from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
from mmcv.runner import build_optimizer
import pytorch_lightning as L
from scipy.spatial.transform import Rotation as R

from models.model_wo_sd import OV9D
import utils.logging as logging
from utils.losses import angular_distance, L1_reg_nocs_loss, CrossEntropyNocsMapLoss
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
        self.criterion_L1 = SmoothL1Loss(beta=0.1)
        self.criterion_CE = CrossEntropyNocsMapLoss(reduction='sum', weight=None)

        self.total_val_steps = 0
        self.xyz_bin = args.nocs_bin

        x_cls_grid_center = (torch.arange(self.xyz_bin) + 0.5) / self.xyz_bin
        y_cls_grid_center = (torch.arange(self.xyz_bin) + 0.5) / self.xyz_bin
        z_cls_grid_center = (torch.arange(self.xyz_bin) + 0.5) / self.xyz_bin

        self.register_buffer("x_grid_center", torch.cat([x_cls_grid_center, torch.tensor([0])]).to(torch.float32))
        self.register_buffer("y_grid_center", torch.cat([y_cls_grid_center, torch.tensor([0])]).to(torch.float32))
        self.register_buffer("z_grid_center", torch.cat([z_cls_grid_center, torch.tensor([0])]).to(torch.float32))

    
    def _step(self, batch, prex, batch_idx):
        gt_rgb = batch['image']
        # input_rgb = batch['input_image']
        mask = batch['mask'].to(bool)
        mask_resized = batch['mask_resized'].to(bool)
        # feat_3d = batch['3d_feat']
        feat_2d_bp = batch['pcl_model'] # (b, 480, 480, 3)
        roi_coord_2d = batch['roi_coord_2d'] # (b, 2, 480, 480)
        bbox_wh = batch['bbox_size'] # (b)
        bbox_center = batch['bbox_center'] # (b, 2)
        ratio = batch['resize_ratio'].squeeze(-1) # (b)
        nocs = batch['nocs'].permute(0, 2, 3, 1) # (b, 480, 480, 3)
        nocs_resized = batch['nocs_resized'].permute(0, 2, 3, 1) # (b, 35, 35, 3)
        gt_xyz_bin = batch['gt_xyz_bin'] # (b, 3, 480, 480)
        dis_sym = batch['dis_sym']
        con_sym = batch['con_sym']
        gt_r = batch['gt_r'] # (b, 3, 3)
        gt_t = batch['gt_t'] # (b, 3)
        cams = batch['cam'] # (b, 3, 3)
        gt_trans_ratio = batch["gt_trans_ratio"] # (B, 3)

        gt_model_center = batch['kps_3d_center'] # (b, 3)
        gt_model_size = batch['kps_3d_dig'] # (b, 1)
        gt_coord_2d = batch['gt_coord_2d'] # (b, 480, 480, h)

        gt_nocs = nocs_resized if self.args.low_res_sup else nocs
        input_MASK = mask_resized if self.args.low_res_sup else mask

        translation_ratio = 100
        gt_t = gt_t / translation_ratio

        b, _, h, w = batch['image'].shape

        # normalize 3d points
        feat_2d_bp_normed = []
        for i in range(b):
            feat_2d_bp_normed.append(normalize_3d_points(feat_2d_bp[i]))
        feat_2d_bp_normed = torch.stack(feat_2d_bp_normed, dim=0).to(feat_2d_bp)

        preds = self.model(gt_rgb, feat_2d_bp_normed, mask, roi_coord_2d, class_ids=batch['class_id'])
        pred_nocs_feat, pred_nocs_offset = preds['pred_nocs_feat'], preds['pred_nocs_offset'] # (b, c, h, w), (b, 3, h ,w)
        pred_nocs_feat_ori_size = None
        if 'pred_nocs_ori_size' in preds:
            pred_nocs_feat_ori_size = preds['pred_nocs_ori_size']
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

        # nocs loss
        loss_classification, loss_regression = None, None
        if self.args.nocs_type == 'L1':
            loss_o = L1_reg_nocs_loss(b, pred_nocs_feat.permute(0, 2, 3, 1), gt_nocs, mask_resized, dis_sym, con_sym, self.criterion_L1)
            pred_nocs = pred_nocs_feat.permute(0, 2, 3, 1)
            # loss_o = nn.L1Loss(reduction='sum')(pred_nocs*input_MASK.unsqueeze(-1), gt_nocs*input_MASK.unsqueeze(-1)) / input_MASK.sum().float().clamp(min=1.0)
        elif self.args.nocs_type == 'CE': # classification + regression
            out_x, out_y, out_z = torch.split(pred_nocs_feat, pred_nocs_feat.shape[1] // 3, dim=1) # (b, xyz_bin, h, w)
            gt_xyz_bin = gt_xyz_bin.long() # (b, 3, h, w)
            loss_coord_x = self.criterion_CE(out_x * input_MASK[:, None, :, :], gt_xyz_bin[:, 0] * input_MASK.long(), input_MASK) / input_MASK.sum().float().clamp(min=1.0)
            loss_coord_y = self.criterion_CE(out_y * input_MASK[:, None, :, :], gt_xyz_bin[:, 1] * input_MASK.long(), input_MASK) / input_MASK.sum().float().clamp(min=1.0)
            loss_coord_z = self.criterion_CE(out_z * input_MASK[:, None, :, :], gt_xyz_bin[:, 2] * input_MASK.long(), input_MASK) / input_MASK.sum().float().clamp(min=1.0)
            loss_classification = loss_coord_x + loss_coord_y + loss_coord_z

            pred_nocs = torch.zeros_like(batch['nocs']).to(batch['nocs']) # (b, 3, h, w)
            x_grid = self.x_grid_center.unsqueeze(0).unsqueeze(1).unsqueeze(1).expand(b, h, w, -1) # (b, h, w, xyz_bin)
            y_grid = self.y_grid_center.unsqueeze(0).unsqueeze(1).unsqueeze(1).expand(b, h, w, -1) # (b, h, w, xyz_bin)
            z_grid = self.z_grid_center.unsqueeze(0).unsqueeze(1).unsqueeze(1).expand(b, h, w, -1) # (b, h, w, xyz_bin)
            for b_i in range(b):
                pred_nocs[b_i][0] = x_grid[b_i][torch.arange(h)[:, None], torch.arange(w), gt_xyz_bin[b_i, 0]]
                pred_nocs[b_i][1] = y_grid[b_i][torch.arange(h)[:, None], torch.arange(w), gt_xyz_bin[b_i, 1]]
                pred_nocs[b_i][2] = z_grid[b_i][torch.arange(h)[:, None], torch.arange(w), gt_xyz_bin[b_i, 2]]
            pred_nocs = ((pred_nocs + pred_nocs_offset)).permute(0, 2, 3, 1)
            loss_regression = L1_reg_nocs_loss(b, pred_nocs, nocs, input_MASK, dis_sym, con_sym, self.criterion_L1)
            loss_o = 0.1*loss_classification + loss_regression
        else:
            raise ValueError("The nocs type should be either CE of L1.")
        
        # self-supervision loss
        pred_nocs_ori_size = pred_nocs_feat_ori_size.permute(0, 2, 3, 1) if pred_nocs_feat_ori_size is not None else pred_nocs
        pred_pcl_m = (pred_nocs_ori_size - 0.5) * gt_model_size[:, :, None, None] + gt_model_center.reshape(b, 1, 1, 3)
        pred_pcl_c = gt_r @ pred_pcl_m.reshape(b, -1, 3).permute(0, 2, 1) + gt_t.unsqueeze(-1)*translation_ratio
        pred_pcl_2d = cams @ pred_pcl_c
        pred_pcl_2d = pred_pcl_2d[:, :2, :] / pred_pcl_2d[:, 2:3, :]
        gt_pcl_2d = gt_coord_2d
        loss_self_suv = (nn.MSELoss(reduction='none')(pred_pcl_2d.reshape(b, 2, h, w).permute(0, 2 ,3 ,1), gt_pcl_2d)*mask.unsqueeze(-1)).sum() ** 0.5 / mask.sum().float().clamp(min=1.0)

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
        loss_total = 2*loss_o + 1e-4*loss_self_suv + loss_rot + 0.1*loss_trans + 0.1*loss_bind + loss_centroid + 0.1*loss_z
        # loss_total = 2*loss_o + 1e-4*loss_self_suv

        self.nocs_loss.update(loss_o.detach().item(), gt_rgb.size(0))

        loss_dict = {
            'loss': loss_total,
            'nocs loss': loss_o.item(),
            'self supervision loss': loss_self_suv.item(),
            'rotation loss': loss_rot.item(),
            'translation loss': loss_trans.item(),
            'Rt loss': loss_bind.item(),
            'centroid loss': loss_centroid.item(),
            'z loss': loss_z.item()
        }
        if loss_classification is not None:
            loss_dict.update({'classification loss': loss_classification.item()})
        if loss_regression is not None:
            loss_dict.update({'regression loss': loss_regression.item()})

        # self.log(f'{prex}/loss', loss_o, on_step=True, on_epoch=True, prog_bar=True)
        self.log_step(loss_dict, prex)

        if prex == "train":
            if 0 == self.trainer.global_step % 20 and (self.trainer.local_rank == 0):
                output_vis = self.vis_images(batch, gt_nocs, nocs, pred_nocs*mask_resized.unsqueeze(-1), pred_nocs_ori_size*mask.unsqueeze(-1), pred_ego_rot, pred_trans*translation_ratio)
                for key, value in output_vis.items():
                    imgs = [np.concatenate([img for img in value],axis=0)]
                    self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
        else:
            if 0 == self.total_val_steps % 20:
                output_vis = self.vis_images(batch, gt_nocs, nocs, pred_nocs*mask_resized.unsqueeze(-1), pred_nocs_ori_size*mask.unsqueeze(-1), pred_ego_rot, pred_trans*translation_ratio)
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
    

    def vis_images(self, batch, gt_nocs, nocs_ori, pred_nocs_vis, pred_nocs_ori_size_vis, pred_ego_rot, pred_trans):
        outputs = {}
        B, C, H, W = batch['image'].shape

        gt_rgb = batch['image'].permute(0, 2, 3, 1).detach().cpu().numpy()
        gt_nocs = gt_nocs.detach().cpu().numpy()
        nocs_ori = nocs_ori.detach().cpu().numpy()
        pred_nocs = pred_nocs_vis.detach().cpu().numpy()
        pred_nocs_ori_size = pred_nocs_ori_size_vis.detach().cpu().numpy()

        raw_scene = batch['raw_scene'].permute(0, 2, 3, 1).detach().cpu().numpy() # (b, 480, 640, 3), np.uint8
        kps_3d_m = batch['kps_3d_m'].detach().cpu().numpy() # (B, 9, 3)
        cams = batch['cam'].detach().cpu().numpy()

        gt_r = batch['gt_r'].detach().cpu().numpy()
        gt_t = batch['gt_t'].detach().cpu().numpy()
        pred_r = pred_ego_rot.detach().cpu().numpy()
        pred_t = pred_trans.detach().cpu().numpy()

        gt_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, gt_r, gt_t)
        pred_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, pred_r, pred_t)

        outputs.update({"gt rgb":gt_rgb, "gt nocs":gt_nocs, "gt nocs ori": nocs_ori, "pred nocs":pred_nocs, "pred nocs ori": pred_nocs_ori_size, "gt pose":gt_pose, "pred pose":pred_pose})

        # outputs.update({"gt rgb":gt_rgb, "gt nocs":gt_nocs, "pred nocs":pred_nocs})

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