import numpy as np
import torch

from .pose_utils import allo_to_ego_mat_torch, allocentric_to_egocentric


def pose_from_pred_centroid_z(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-8,
    is_allo=True,
    z_type="REL",
):
    """for train
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG

    Returns:

    """
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()
    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "ABS":
        z = pred_z_vals
    elif z_type == "REL":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    # NOTE: z must be [B,1]
    translation = torch.cat(
        [z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0], z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1], z],
        dim=1,
    )

    assert pred_rots.ndim == 3 and pred_rots.shape[-1] == 3 # Nx3x3
    if is_allo:
        rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
    else:
        rot_ego = pred_rots
    return rot_ego, translation


def pose_from_predictions_test(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """NOTE: for test, non-differentiable"""
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()

    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "ABS":
        z = pred_z_vals
    elif z_type == "REL":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    translation = torch.cat(
        [z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0], z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1], z],
        dim=1,
    )

    # rot mat
    assert pred_rots.shape[-1] == 3 and pred_rots.ndim == 3
    pred_rots = pred_rots.detach().cpu().numpy()
    ego_rot_preds = np.zeros_like(pred_rots)
    for i in range(pred_rots.shape[0]):
        if is_allo:
            cur_ego_mat = allocentric_to_egocentric(
                np.hstack([pred_rots[i], translation[i].detach().cpu().numpy().reshape(3, 1)]),
                src_type="mat",
                dst_type="mat",
            )[:3, :3]
        else:
            cur_ego_mat = pred_rots[i]
        ego_rot_preds[i] = cur_ego_mat
    return torch.from_numpy(ego_rot_preds), translation


def pose_from_predictions_train(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """for train
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG

    Returns:

    """
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()
    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "ABS":
        z = pred_z_vals
    elif z_type == "REL":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    # NOTE: z must be [B,1]
    translation = torch.cat(
        [z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0], z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1], z],
        dim=1,
    )

    assert pred_rots.ndim == 3 and pred_rots.shape[-1] == 3 # Nx3x3
    if is_allo:
        rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
    else:
        rot_ego = pred_rots
    return rot_ego, translation