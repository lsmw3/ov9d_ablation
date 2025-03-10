import torch


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