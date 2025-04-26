from typing import Dict
import os
import re
import cv2
import math
import json
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


def transform_pts_batch(pts, R, t=None):
    """
    Args:
        pts: (B,P,3)
        R: (B,3,3)
        t: (B,3,1)

    Returns:

    """
    bs = R.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bs, n_pts, 3)
    if t is not None:
        assert t.shape[0] == bs

    pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
    if t is not None:
        pts_transformed += t.view(bs, 1, 3, 1)
    return pts_transformed.squeeze(-1)  # (B, P, 3)


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

    
    def _step(self, batch, prex, batch_idx, log=True):
        gt_rgb = batch['image']
        # input_rgb = batch['input_image']
        mask = batch['mask'].to(bool)
        mask_resized = batch['mask_resized'].to(bool)
        # feat_3d = batch['3d_feat']
        # feat_2d_bp = batch['pcl_model'] # (b, 480, 480, 3)
        roi_coord_2d = batch['roi_coord_2d'] # (b, 2, 480, 480)
        bbox_wh = batch['bbox_size'] # (b)
        bbox_center = batch['bbox_center'] # (b, 2)
        ratio = batch['resize_ratio'].squeeze(-1) # (b)
        nocs = batch['nocs'].permute(0, 2, 3, 1) # (b, 480, 480, 3)
        nocs_resized = batch['nocs_resized'].permute(0, 2, 3, 1) # (b, 35, 35, 3)
        nocs_mask = batch['nocs_mask'].to(bool) # (b, 35, 35)
        dis_sym = batch['dis_sym']
        con_sym = batch['con_sym']
        gt_r = batch['gt_r'] # (b, 3, 3)
        gt_t = batch['gt_t'] # (b, 3)
        cams = batch['cam'] # (b, 3, 3)
        gt_trans_ratio = batch["gt_trans_ratio"] # (B, 3)

        gt_kps_3d = batch['kps_3d_m'] # (b, 9, 3)
        gt_model_center = batch['kps_3d_center'] # (b, 3)
        gt_model_size = batch['kps_3d_dig'] # (b, 1)
        gt_coord_2d = batch['gt_coord_2d'] # (b, 480, 480, h)

        gt_nocs = nocs_resized
        input_MASK = mask_resized

        translation_ratio = 1
        gt_t = gt_t / translation_ratio

        b, _, h, w = batch['image'].shape

        # # normalize 3d points
        # feat_2d_bp_normed = []
        # for i in range(b):
        #     feat_2d_bp_normed.append(normalize_3d_points(feat_2d_bp[i]))
        # feat_2d_bp_normed = torch.stack(feat_2d_bp_normed, dim=0).to(feat_2d_bp)

        preds = self.model(gt_rgb, mask, roi_coord_2d)

        if self.args.decode_rt:
            pred_r, pred_t, pred_dims = preds['pred_r'], preds['pred_t'], preds['pred_dims']
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
            pred_center_2d = torch.stack(
                [
                    (pred_t[:, :2][:, 0] * bbox_wh) + bbox_center[:, 0],
                    (pred_t[:, :2][:, 1] * bbox_wh) + bbox_center[:, 1],
                ],
                dim=1,
            )
            self.model_output = {'pred_ego_rot': pred_ego_rot, 'pred_trans': pred_trans, 'pred_dims': pred_dims, 'pred_center_2d': pred_center_2d}
            # Size loss
            kpts_m = torch.zeros([b, 8, 3]).to(gt_kps_3d)
            l, w, h = pred_dims[:, 0].unsqueeze(1), pred_dims[:, 1].unsqueeze(1), pred_dims[:, 2].unsqueeze(1)
            kpts_m[:, [0, 1, 2, 3], 0], kpts_m[:, [4, 5, 6, 7], 0] = -l/2, l/2
            kpts_m[:, [0, 1, 4, 5], 1], kpts_m[:, [2, 3, 6, 7], 1] = -w/2, w/2
            kpts_m[:, [0, 2, 4, 6], 2], kpts_m[:, [1, 3, 5, 7], 2] = -h/2, h/2
            kpts_pred = transform_pts_batch(kpts_m, gt_r, gt_t)
            kpts_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
            loss_sz = nn.L1Loss(reduction="mean")(kpts_pred, kpts_tgt)
            if torch.isnan(loss_sz):
                print("loss size is nan")
            # self.model_output.update({'pred_pts_3d': kpts_pred})
            # points matching loss
            kps_pred = transform_pts_batch(gt_kps_3d[:, 1:, :], pred_ego_rot, pred_trans)
            kps_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
            loss_pm = nn.L1Loss(reduction="mean")(kps_pred, kps_tgt)
            if torch.isnan(loss_pm):
                print("loss pm is nan")

            # points-bind loss
            bbox_3d_pred = transform_pts_batch(kpts_m, pred_ego_rot, pred_trans)
            bbox_3d_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
            loss_pts_bind = nn.L1Loss(reduction="mean")(bbox_3d_pred, bbox_3d_tgt)
            
            self.model_output.update({'pred_pts_3d': bbox_3d_pred, 'gt_pts_3d': bbox_3d_tgt})

            # rotation loss
            loss_rot = angular_distance(pred_ego_rot, gt_r)
            if torch.isnan(loss_rot):
                print("loss rot is nan")
            #translation loss
            loss_trans = nn.L1Loss(reduction="mean")(pred_trans, gt_t)
            if torch.isnan(loss_trans):
                print("loss trans is nan")
            # bind loss
            pred_bind = torch.bmm(pred_ego_rot.permute(0, 2, 1), pred_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_r.permute(0, 2, 1), gt_t.view(-1, 3, 1)).view(-1, 3)
            loss_bind = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            if torch.isnan(loss_bind):
                print("loss bind is nan")
            # centroid loss
            loss_centroid = nn.L1Loss(reduction="mean")(pred_t[:, :2], gt_trans_ratio[:, :2])
            if torch.isnan(loss_centroid):
                print("loss centroid is nan")
            # z loss
            loss_z = nn.L1Loss(reduction="mean")(pred_t[:, 2], gt_trans_ratio[:, 2]/translation_ratio)
            if torch.isnan(loss_z):
                print("loss z is nan")

            loss_total = loss_rot + loss_pm + loss_pts_bind + loss_sz + loss_trans + loss_bind + loss_centroid + loss_z
            self.loss_dict = {
                'loss': loss_total,
                'pm loss': loss_pm.item(),
                'size loss': loss_sz.item(),
                'pts bind loss': loss_pts_bind.item(),
                'rotation loss': loss_rot.item(),
                'translation loss': loss_trans.item(),
                'Rt loss': loss_bind.item(),
                'centroid loss': loss_centroid.item(),
                'z loss': loss_z.item()
            }

            if log:
                self.log_step(self.loss_dict, prex)

                if prex == "train":
                    if 0 == self.trainer.global_step % 500 and (self.trainer.local_rank == 0):
                        output_vis = self.vis_pose(batch, pred_ego_rot, pred_trans*translation_ratio, pred_dims)
                        for key, value in output_vis.items():
                            imgs = [np.concatenate([img for img in value],axis=0)]
                            self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
                else:
                    if 0 == self.total_val_steps % 200:
                        output_vis = self.vis_pose(batch, pred_ego_rot, pred_trans*translation_ratio, pred_dims)
                        for key, value in output_vis.items():
                            imgs = [np.concatenate([img for img in value],axis=0)]
                            self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)

                    self.total_val_steps += 1

            torch.cuda.empty_cache()

            return loss_total


        pred_nocs_feat, pred_nocs_ori_size = preds['pred_nocs_feat'], preds['pred_nocs_ori_size'].permute(0, 2, 3, 1) # (b, 3, h, w), (b, H, W, 3)
        pred_r, pred_t, pred_dims = preds['pred_r'], preds['pred_t'], preds['pred_dims']

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

        pred_center_2d = torch.stack(
            [
                (pred_t[:, :2][:, 0] * bbox_wh) + bbox_center[:, 0],
                (pred_t[:, :2][:, 1] * bbox_wh) + bbox_center[:, 1],
            ],
            dim=1,
        )
        self.model_output = {'pred_ego_rot': pred_ego_rot, 'pred_trans': pred_trans, 'pred_dims': pred_dims, 'pred_center_2d': pred_center_2d}

        # nocs loss
        # loss_o = L1_reg_nocs_loss(b, pred_nocs_feat.permute(0, 2, 3, 1), gt_nocs, nocs_mask, dis_sym, con_sym, self.criterion_L1)
        loss_o = self.criterion_L1(pred_nocs_feat.permute(0, 2, 3, 1)[nocs_mask], gt_nocs[nocs_mask]) + 0.5 * self.criterion_L1(pred_nocs_feat.permute(0, 2, 3, 1)[~mask_resized], gt_nocs[~mask_resized])
        pred_nocs = pred_nocs_feat.permute(0, 2, 3, 1)
        
        # # self-supervision loss
        # pred_nocs_ori_size = pred_nocs_feat_ori_size.permute(0, 2, 3, 1) if pred_nocs_feat_ori_size is not None else pred_nocs
        # pred_pcl_m = (pred_nocs_ori_size - 0.5) * gt_model_size[:, :, None, None] + gt_model_center.reshape(b, 1, 1, 3)
        # pred_pcl_c = gt_r @ pred_pcl_m.reshape(b, -1, 3).permute(0, 2, 1) + gt_t.unsqueeze(-1)*translation_ratio
        # pred_pcl_2d = cams @ pred_pcl_c
        # pred_pcl_2d = pred_pcl_2d[:, :2, :] / pred_pcl_2d[:, 2:3, :]
        # gt_pcl_2d = gt_coord_2d
        # loss_self_suv = (nn.L1Loss(reduction='none')(pred_pcl_2d.reshape(b, 2, h, w).permute(0, 2 ,3 ,1), gt_pcl_2d)*mask.unsqueeze(-1)).sum() / mask.sum().float().clamp(min=1.0)

        # loss_self_suv = self.rescale_loss(loss_self_suv)

        # size loss
        kpts_m = torch.zeros([b, 8, 3]).to(gt_kps_3d)
        l, w, h = pred_dims[:, 0].unsqueeze(1), pred_dims[:, 1].unsqueeze(1), pred_dims[:, 2].unsqueeze(1)
        kpts_m[:, [0, 1, 2, 3], 0], kpts_m[:, [4, 5, 6, 7], 0] = -l/2, l/2
        kpts_m[:, [0, 1, 4, 5], 1], kpts_m[:, [2, 3, 6, 7], 1] = -w/2, w/2
        kpts_m[:, [0, 2, 4, 6], 2], kpts_m[:, [1, 3, 5, 7], 2] = -h/2, h/2
        kpts_pred = transform_pts_batch(kpts_m, gt_r, gt_t)
        kpts_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
        loss_sz = nn.L1Loss(reduction="mean")(kpts_pred, kpts_tgt)
        # self.model_output.update({'pred_pts_3d': kpts_pred})

        # points matching loss
        kps_pred = transform_pts_batch(gt_kps_3d[:, 1:, :], pred_ego_rot, pred_trans)
        kps_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
        loss_pm = nn.L1Loss(reduction="mean")(kps_pred, kps_tgt)
        # loss_pm = self.rescale_loss(loss_pm)

        # points-bind loss
        bbox_3d_pred = transform_pts_batch(kpts_m, pred_ego_rot, pred_trans)
        bbox_3d_tgt = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
        loss_pts_bind = nn.L1Loss(reduction="mean")(bbox_3d_pred, bbox_3d_tgt)

        self.model_output.update({'pred_pts_3d': bbox_3d_pred, 'gt_pts_3d': bbox_3d_tgt})

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
        loss_total = loss_o + loss_pm + loss_pts_bind + loss_rot + loss_trans + loss_bind + loss_centroid + loss_z + loss_sz

        self.nocs_loss.update(loss_o.detach().item(), gt_rgb.size(0))

        self.loss_dict = {
            'loss': loss_total,
            'nocs loss': loss_o.item(),
            # 'self supervision loss': loss_self_suv.item(),
            'size loss': loss_sz.item(),
            'pm loss': loss_pm.item(),
            'pts bind loss': loss_pts_bind.item(),
            'rotation loss': loss_rot.item(),
            'translation loss': loss_trans.item(),
            'Rt loss': loss_bind.item(),
            'centroid loss': loss_centroid.item(),
            'z loss': loss_z.item()
        }

        # self.log(f'{prex}/loss', loss_o, on_step=True, on_epoch=True, prog_bar=True)
        if log:
            self.log_step(self.loss_dict, prex)

            if prex == "train":
                if 0 == self.trainer.global_step % 1000 and (self.trainer.local_rank == 0):
                    output_vis = self.vis_images(batch, gt_nocs, nocs, pred_nocs*mask_resized.unsqueeze(-1), pred_nocs_ori_size*mask.unsqueeze(-1), mask_resized, nocs_mask, pred_ego_rot, pred_trans*translation_ratio, pred_dims)
                    for key, value in output_vis.items():
                        imgs = [np.concatenate([img for img in value],axis=0)]
                        self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
            else:
                if 0 == self.total_val_steps % 500:
                    output_vis = self.vis_images(batch, gt_nocs, nocs, pred_nocs*mask_resized.unsqueeze(-1), pred_nocs_ori_size*mask.unsqueeze(-1), mask_resized, nocs_mask, pred_ego_rot, pred_trans*translation_ratio, pred_dims)
                    for key, value in output_vis.items():
                        imgs = [np.concatenate([img for img in value],axis=0)]
                        self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)

                self.total_val_steps += 1

        torch.cuda.empty_cache()

        return loss_total


    def clean_numbered_files(self, folder_path, pattern):
        """
        Deletes all but the two most recent (largest number) files in a folder.

        :param folder_path: Path to the folder containing files.
        :param pattern: Regex pattern to match filenames and extract numbers.
        """
        regex = re.compile(pattern)
        
        # Find all matching files and extract their numbers
        files = []
        for filename in os.listdir(folder_path):
            match = regex.match(filename)
            if match:
                file_number = int(match.group(1))  # Extract the number
                files.append((file_number, os.path.join(folder_path, filename)))

        # Sort files by extracted number in descending order
        files.sort(reverse=True, key=lambda x: x[0])

        # If more than 2 files exist, delete all except the last two
        if len(files) > 20:
            for _, file_path in files[2:]:  # Keep the first two, delete the rest
                os.remove(file_path)
    

    def rescale_loss(self, loss):
        if loss > 1000:
            loss = loss * 1e-4
        elif 1000 >= loss > 100:
            loss = loss * 1e-3
        elif 100 >= loss > 10:
            loss = loss * 1e-2
        elif 10 >= loss > 1:
            loss = loss * 0.1
        else:
            loss = loss
        
        return loss
    

    def log_step(self, objectives: Dict[str, torch.Tensor], prex: str):
        for key, value in objectives.items():
            self.log(
                f"{prex}/{key}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )


    def vis_images(self, batch, gt_nocs, nocs_ori, pred_nocs_vis, pred_nocs_ori_size_vis, mask, nocs_mask, pred_ego_rot, pred_trans, pred_dims):
        outputs = {}
        B, C, H, W = batch['image'].shape

        mask = mask.detach().cpu().numpy()
        nocs_mask = nocs_mask.detach().cpu().numpy()
        gt_rgb = batch['image'].permute(0, 2, 3, 1).detach().cpu().numpy()
        gt_nocs = gt_nocs.detach().cpu().numpy()
        nocs_ori = nocs_ori.detach().cpu().numpy()
        pred_nocs = pred_nocs_vis.detach().cpu().numpy()
        pred_nocs_ori_size = pred_nocs_ori_size_vis.detach().cpu().numpy()

        # raw_scene = batch['raw_scene'].permute(0, 2, 3, 1).detach().cpu().numpy() # (b, 480, 640, 3), np.uint8
        kps_3d_m = batch['kps_3d_m'].detach().cpu().numpy() # (B, 9, 3)
        cams = batch['cam'].detach().cpu().numpy()

        gt_r = batch['gt_r'].detach().cpu().numpy()
        gt_t = batch['gt_t'].detach().cpu().numpy()
        pred_r = pred_ego_rot.detach().cpu().numpy()
        pred_t = pred_trans.detach().cpu().numpy()

        kpts_m = np.zeros([B, 9, 3]).astype(np.float32)
        l, w, h = pred_dims[:, 0].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 1].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 2].unsqueeze(1).detach().cpu().numpy()
        kpts_m[:, [1, 2, 3, 4], 0], kpts_m[:, [5, 6, 7, 8], 0] = -l/2, l/2
        kpts_m[:, [1, 2, 5, 6], 1], kpts_m[:, [3, 4, 7, 8], 1] = -w/2, w/2
        kpts_m[:, [1, 3, 5, 7], 2], kpts_m[:, [2, 4, 6, 8], 2] = -h/2, h/2

        # gt_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, gt_r, gt_t)
        # pred_pose = self.vis_pose_on_img(raw_scene, kpts_m, cams, pred_r, pred_t)

        if isinstance(batch['raw_scene'], torch.Tensor):
            raw_scene = batch['raw_scene'].permute(0, 2, 3, 1).detach().cpu().numpy() # (b, 480, 640, 3), np.uint8
            gt_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, gt_r, gt_t)
            pred_pose = self.vis_pose_on_img(raw_scene, kpts_m, cams, pred_r, pred_t)
        else:
            assert isinstance(batch['raw_scene'], list)
            gt_pose_list, pred_pose_list = [], []
            raw_scene = batch['raw_scene']
            for i in range(B):
                gt_pose_single = self.vis_pose_on_img(raw_scene[i].permute(1, 2, 0).unsqueeze(0).detach().cpu().numpy(), kps_3d_m[i][None], cams[i][None], gt_r[i][None], gt_t[i][None])[0]
                pred_pose_single = self.vis_pose_on_img(raw_scene[i].permute(1, 2, 0).unsqueeze(0).detach().cpu().numpy(), kpts_m[i][None], cams[i][None], pred_r[i][None], pred_t[i][None])[0]

                gt_pose_list.append(cv2.resize(gt_pose_single, (1440, 1440), cv2.INTER_LINEAR))
                pred_pose_list.append(cv2.resize(pred_pose_single, (1440, 1440), cv2.INTER_LINEAR))

            gt_pose = np.array(gt_pose_list)
            pred_pose = np.array(pred_pose_list)

        outputs.update({"gt rgb":gt_rgb, "gt nocs":gt_nocs, "gt nocs ori": nocs_ori, "pred nocs":pred_nocs, "pred nocs ori": pred_nocs_ori_size, "mask": mask, "nocs mask": nocs_mask, "gt pose":gt_pose, "pred pose":pred_pose})

        # outputs.update({"gt rgb":gt_rgb, "gt nocs":gt_nocs, "pred nocs":pred_nocs})

        return outputs

    
    def vis_pose(self, batch, pred_ego_rot, pred_trans, pred_dims):
        outputs = {}
        B, C, H, W = batch['image'].shape

        gt_rgb = batch['image'].permute(0, 2, 3, 1).detach().cpu().numpy()
        kps_3d_m = batch['kps_3d_m'].detach().cpu().numpy() # (B, 9, 3)
        cams = batch['cam'].detach().cpu().numpy()

        gt_r = batch['gt_r'].detach().cpu().numpy()
        gt_t = batch['gt_t'].detach().cpu().numpy()
        pred_r = pred_ego_rot.detach().cpu().numpy()
        pred_t = pred_trans.detach().cpu().numpy()

        kpts_m = np.zeros([B, 9, 3]).astype(np.float32)
        l, w, h = pred_dims[:, 0].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 1].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 2].unsqueeze(1).detach().cpu().numpy()
        kpts_m[:, [1, 2, 3, 4], 0], kpts_m[:, [5, 6, 7, 8], 0] = -l/2, l/2
        kpts_m[:, [1, 2, 5, 6], 1], kpts_m[:, [3, 4, 7, 8], 1] = -w/2, w/2
        kpts_m[:, [1, 3, 5, 7], 2], kpts_m[:, [2, 4, 6, 8], 2] = -h/2, h/2

        if isinstance(batch['raw_scene'], torch.Tensor):
            raw_scene = batch['raw_scene'].permute(0, 2, 3, 1).detach().cpu().numpy() # (b, 480, 640, 3), np.uint8
            gt_pose = self.vis_pose_on_img(raw_scene, kps_3d_m, cams, gt_r, gt_t)
            pred_pose = self.vis_pose_on_img(raw_scene, kpts_m, cams, pred_r, pred_t)
        else:
            assert isinstance(batch['raw_scene'], list)
            gt_pose_list, pred_pose_list = [], []
            raw_scene = batch['raw_scene']
            for i in range(B):
                gt_pose_single = self.vis_pose_on_img(raw_scene[i].permute(1, 2, 0).unsqueeze(0).detach().cpu().numpy(), kps_3d_m[i][None], cams[i][None], gt_r[i][None], gt_t[i][None])[0]
                pred_pose_single = self.vis_pose_on_img(raw_scene[i].permute(1, 2, 0).unsqueeze(0).detach().cpu().numpy(), kpts_m[i][None], cams[i][None], pred_r[i][None], pred_t[i][None])[0]

                gt_pose_list.append(cv2.resize(gt_pose_single, (1440, 1440), cv2.INTER_LINEAR))
                pred_pose_list.append(cv2.resize(pred_pose_single, (1440, 1440), cv2.INTER_LINEAR))

            gt_pose = np.array(gt_pose_list)
            pred_pose = np.array(pred_pose_list)

        outputs.update({"gt rgb":gt_rgb, "gt pose":gt_pose, "pred pose":pred_pose})

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
    

    def forward(self, batch):
        return self._step(batch, 'inference', None, False)
    
    
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
        
        def lr_lambda(step):
            initial_lr = self.args.lr
            min_lr = self.args.min_lr
            total_steps = 20000
            if step >= total_steps:
                return min_lr / initial_lr
            return min_lr / initial_lr + (1 - min_lr / initial_lr) * (1 + math.cos(math.pi * step / total_steps)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }