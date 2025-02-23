import numpy as np
import torch
import torch.nn as nn
from torch.nn import SmoothL1Loss, L1Loss
from mmcv.runner import build_optimizer
import pytorch_lightning as L
from scipy.spatial.transform import Rotation as R

from models.model_wo_sd import OV9D
import utils.logging as logging


class CustomTrainer(L.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.model = OV9D(args)
        self.nocs_loss = logging.AverageMeter()
        self.criterion = SmoothL1Loss(beta=0.1)

    
    def _step(self, batch, prex, batch_idx):
        input_RGB = batch['image']
        input_MASK = batch['mask'].to(bool)
        nocs = batch['nocs'].permute(0, 2, 3, 1)
        dis_sym = batch['dis_sym']
        con_sym = batch['con_sym']

        preds = self.model(input_RGB, class_ids=batch['class_id'])
        pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)
        
        pred_nocs_list, gt_nocs_list = [], []
        for b in range(batch['image'].shape[0]):
            curr_pred_nocs = pred_nocs[b]
            curr_gt_nocs = nocs[b]
            curr_mask = input_MASK[b]
            curr_pred_nocs = curr_pred_nocs[curr_mask]
            curr_gt_nocs = curr_gt_nocs[curr_mask]
            curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
            # discrete symmetry
            curr_dis_sym = dis_sym[b]
            dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
            curr_dis_sym = curr_dis_sym[dis_sym_flag]
            aug_pcl_m = torch.stack([curr_pcl_m], dim=0)
            for sym in curr_dis_sym:
                rot, t = sym[0:3, 0:3], sym[0:3, 3]
                rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
                aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0)
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
                loss = list(map(lambda gt_nocs: self.criterion(curr_pred_nocs, gt_nocs), curr_gt_nocs_set))
                min_idx = torch.argmin(torch.tensor(loss))
            curr_gt_nocs = curr_gt_nocs_set[min_idx]
            
            pred_nocs_list.append(curr_pred_nocs)
            gt_nocs_list.append(curr_gt_nocs)

        loss_o = self.criterion(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list))

        self.nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))

        self.log(f'{prex}/loss', loss_o, on_step=True, on_epoch=True, prog_bar=True)

        return loss_o

    
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train', batch_idx)
    

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val', batch_idx)
    
    
    def configure_optimizers(self):
        # Initialize the optimizer configuration
        optimizer_cfg = {
            'type': 'AdamW',
            'lr': self.args.max_lr,
            'betas': (0.9, 0.999),
            'weight_decay': self.args.weight_decay,
            'paramwise_cfg': {
                'layer_decay_rate': self.args.layer_decay,
                'no_decay_names': ['relative_position_bias_table', 'rpe_mlp', 'logit_scale']
            }
        }

        # Build the optimizer using MMCV's build_optimizer function
        optimizer = build_optimizer(self.model, optimizer_cfg)

        # Return the optimizer
        return optimizer