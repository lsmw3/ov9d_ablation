import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation as R


def angular_distance(r1, r2, reduction="mean"):
    """https://math.stackexchange.com/questions/90081/quaternion-distance
    https.

    ://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tool
    s.py.

    1 - <q1, q2>^2  <==> (1-cos(theta)) / 2
    """
    assert r1.shape == r2.shape
    if r1.shape[-1] == 4:
        return angular_distance_quat(r1, r2, reduction=reduction)
    else:
        return angular_distance_rot(r1, r2, reduction=reduction)


def angular_distance_quat(pred_q, gt_q, reduction="mean"):
    dist = 1 - torch.pow(torch.bmm(pred_q.view(-1, 1, 4), gt_q.view(-1, 4, 1)), 2)
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def angular_distance_rot(m1, m2, reduction="mean"):
    m = torch.bmm(m1, m2.transpose(1, 2))  # b*3*3
    m_trace = torch.einsum("bii->b", m)  # batch trace
    cos = (m_trace - 1) / 2  # [-1, 1]
    # eps = 1e-6
    # cos = torch.clamp(cos, -1+eps, 1-eps)  # avoid nan
    # theta = torch.acos(cos)
    dist = (1 - cos) / 2  # [0, 1]
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def rot_l2_loss(m1, m2):
    error = torch.pow(m1 - m2, 2).mean()  # batch
    return error


def L1_reg_nocs_loss(bs, pred_nocs, gt_nocs, mask, dis_sym, con_sym, loss_criterion):
    for b in range(bs):
        curr_pred_nocs = pred_nocs[b]  # (480, 480, 3)
        curr_gt_nocs = gt_nocs[b]  # (480, 480, 3)
        curr_mask = mask[b]  # (480, 480)
        
        # Extract masked regions for symmetry handling
        curr_pred_nocs_model_flat = curr_pred_nocs[curr_mask]  # (N, 3) where N is number of masked pixels
        curr_gt_nocs_model_flat = curr_gt_nocs[curr_mask]  # (N, 3)
        
        # Convert NOCS to point cloud for symmetry handling
        curr_pcl_m = curr_gt_nocs_model_flat - 0.5  # NOCS to PCL
        
        # Handle discrete symmetry
        curr_dis_sym = dis_sym[b]
        dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
        curr_dis_sym = curr_dis_sym[dis_sym_flag]
        
        aug_pcl_m = torch.stack([curr_pcl_m], dim=0)  # (1, N, 3)
        for sym in curr_dis_sym:
            rot, t = sym[0:3, 0:3], sym[0:3, 3]
            rot_pcl_m = aug_pcl_m @ rot.T + t.reshape(1, 1, 3)
            aug_pcl_m = torch.cat([aug_pcl_m, rot_pcl_m], dim=0)  # (M, N, 3)
        
        # Handle continuous symmetry
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
        
        # Convert back to NOCS
        curr_gt_nocs_set_flat = aug_pcl_m + 0.5  # (K, N, 3) where K is number of symmetry augmentations
        
        # Find the best symmetry transformation
        with torch.no_grad():
            curr_gt_nocs_set_flat = torch.unbind(curr_gt_nocs_set_flat, dim=0)
            loss = list(map(lambda gt_nocs: loss_criterion(curr_pred_nocs_model_flat, gt_nocs), curr_gt_nocs_set_flat))
            min_idx = torch.argmin(torch.tensor(loss))
            best_gt_nocs_flat = curr_gt_nocs_set_flat[min_idx]  # (N, 3)
        
        # Create spatial versions (keeping original shape)
        # Initialize with original GT NOCS
        best_gt_nocs_spatial = curr_gt_nocs.clone()  # (480, 480, 3)
        
        # Update only the masked regions with the best symmetry-aligned version
        best_gt_nocs_spatial[curr_mask] = best_gt_nocs_flat
        
        # Store results for batch loss computation (maintaining spatial dimensions)
        if b == 0:
            batch_pred_nocs = curr_pred_nocs.unsqueeze(0)  # (1, 480, 480, 3)
            batch_best_gt_nocs = best_gt_nocs_spatial.unsqueeze(0)  # (1, 480, 480, 3)
        else:
            batch_pred_nocs = torch.cat([batch_pred_nocs, curr_pred_nocs.unsqueeze(0)], dim=0)  # (b, 480, 480, 3)
            batch_best_gt_nocs = torch.cat([batch_best_gt_nocs, best_gt_nocs_spatial.unsqueeze(0)], dim=0)  # (b, 480, 480, 3)

    # nocs loss
    loss_o = 2*loss_criterion(batch_pred_nocs[mask], batch_best_gt_nocs[mask]) + 0.5*loss_criterion(batch_pred_nocs[~mask], batch_best_gt_nocs[~mask])

    return loss_o


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "nn criterions don't compute the gradient w.r.t. targets - please "
        "mark these tensors as not requiring gradients"
    )

class CrossEntropyNocsMapLoss(nn.Module):
    def __init__(self, reduction, weight=None, ignore_index=-100):
        super(CrossEntropyNocsMapLoss, self).__init__()
        self.m = nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        if weight is not None:  # bin_size+1
            weight_ = torch.ones(weight)
            weight_[weight - 1] = 0 # bg
            self.loss = nn.NLLLoss(reduction=reduction, weight=weight_, ignore_index=ignore_index)
        else:
            self.loss = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, coor, gt_coor, mask):
        _assert_no_grad(gt_coor)
        log_probs = self.m(coor)
        gt_coor_masked = gt_coor.clone()  
        gt_coor_masked[mask == False] = self.ignore_index
        loss = self.loss(log_probs, gt_coor_masked)
        return loss
