import numpy as np
from scipy.optimize import linear_sum_assignment

def box3d_to_vertices(center, dims, R):
    """
    Convert a 3D bounding box to its 8 corner vertices.

    Args:
        center (array-like, shape=(3,)): box center [x,y,z]
        dims   (array-like, shape=(3,)): box dimensions [length, width, height]
        R      (array-like, shape=(3,3)): rotation matrix
    Returns:
        verts (np.ndarray, shape=(8,3)): world-coordinate corners
    """
    l, w, h = dims
    # Define box in its local frame
    x_c = np.array([[-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
                    [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
                    [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
                    [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2]])
    # Rotate and translate
    verts = (R @ x_c.T).T + center
    return verts


def compute_3d_iou(box1, box2, num_samples=200000):
    """
    Approximate 3D IoU by Monte Carlo sampling inside the union bounding volume.
    This is for educational insight; production code should compute exact polyhedral intersection.

    Args:
        box1, box2: dicts with keys 'center', 'dims', 'R'
        num_samples (int): number of random samples
    Returns:
        iou (float): approximate intersection over union
    """
    # Get vertices
    v1 = box3d_to_vertices(box1['center'], box1['dims'], box1['R'])
    v2 = box3d_to_vertices(box2['center'], box2['dims'], box2['R'])
    # Compute axis-aligned bounding box of the union
    all_pts = np.vstack([v1, v2])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    vol_union_aabb = np.prod(maxs - mins)

    # Sample points uniformly in AABB
    samples = np.random.rand(num_samples, 3) * (maxs - mins) + mins

    # Simple inclusion test assuming convexity: use half-space tests
    # For box1
    def inside(pts, verts):
        # Compute 6 face planes from verts (approx axis-aligned for clipped boxes)
        # Here we skip accurate check; placeholder: both boxes treated as axis-aligned
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        return np.all((pts >= mn) & (pts <= mx), axis=1)

    in1 = inside(samples, v1)
    in2 = inside(samples, v2)

    inter = np.logical_and(in1, in2).sum() / num_samples * vol_union_aabb
    vol1 = in1.sum() / num_samples * vol_union_aabb
    vol2 = in2.sum() / num_samples * vol_union_aabb
    iou = inter / (vol1 + vol2 - inter + 1e-8)
    return iou


def average_precision(recall, precision):
    """
    Compute AP by area under the precision-recall curve (11-point interpolation).
    """
    # Append sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # Make precision monotonically decreasing
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    # Integrate area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap


def compute_pr(gt_boxes, pred_boxes, iou_thresh=0.25, max_dets=100):
    """
    Compute precision and recall for one IoU threshold.
    Args:
        gt_boxes   (list of dict): each dict has 'bbox3D':{'center','dims','R'}
        pred_boxes (list of dict): each dict has 'bbox3D', 'score'
    Returns:
        recall, precision (np.ndarray)
    """
    # Sort preds by descending score
    pred_sorted = sorted(pred_boxes, key=lambda x: -x['score'])[:max_dets]
    TP = np.zeros(len(pred_sorted))
    FP = np.zeros(len(pred_sorted))
    matched = set()
    for i, pred in enumerate(pred_sorted):
        ious = [compute_3d_iou(pred['bbox3D'], gt['bbox3D']) for gt in gt_boxes]
        if not ious:
            FP[i] = 1
            continue
        best_iou = max(ious)
        best_j = np.argmax(ious)
        if best_iou >= iou_thresh and best_j not in matched:
            TP[i] = 1
            matched.add(best_j)
        else:
            FP[i] = 1
    # Cumulative
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    recall = cum_TP / (len(gt_boxes) + 1e-8)
    precision = cum_TP / (cum_TP + cum_FP + 1e-8)
    return recall, precision


def compute_AP_AR(gt, preds):
    """
    Compute standard 3D AP and AR metrics.
    Returns a dict with keys:
      'AP', 'AP15', 'AP25', 'AP50', 
      'APn', 'APm', 'APf',
      'AR1', 'AR10', 'AR100'
    """
    results = {}
    # AP at different IoUs
    for name, thr in [('AP',  None), ('AP15', 0.15), ('AP25', 0.25), ('AP50', 0.50)]:
        # COCO: AP is average over IoUs from 0.05 to 0.5 (if thr is None),
        # here we just compute at single value for simplicity
        iou = thr if thr is not None else 0.25
        recall, precision = compute_pr(gt, preds, iou_thresh=iou, max_dets=100)
        results[name] = average_precision(recall, precision)
    # Depth-based AP
    depths = [gt_inst['bbox3D']['center'][2] for gt_inst in gt]
    if depths:
        depth_vals = np.array(depths)
        bins = {'n': (0,10), 'm': (10,35), 'f': (35, np.inf)}
        for key, (lo, hi) in bins.items():
            sel = [i for i, d in enumerate(depth_vals) if lo <= d < hi]
            gt_bin = [gt[i] for i in sel]
            recall, precision = compute_pr(gt_bin, preds, iou_thresh=0.25, max_dets=100)
            results[f'AP{key}'] = average_precision(recall, precision)
    else:
        for key in ['n','m','f']:
            results[f'AP{key}'] = np.nan
    # AR for different max detections
    for maxd in [1,10,100]:
        recall, precision = compute_pr(gt, preds, iou_thresh=0.25, max_dets=maxd)
        # AR is max recall
        results[f'AR{maxd}'] = recall[-1] if len(recall) else 0.0
    return results


def calculate_nhd(pred_vertices, gt_vertices):
    cost = np.linalg.norm(pred_vertices[:, None, :] - gt_vertices[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    nhd = cost[row, col].sum()
    gt_diag = np.linalg.norm(gt_vertices.max(axis=0) - gt_vertices.min(axis=0))
    return nhd / (gt_diag + 1e-8)


def disentangled_nhd(pred_box, gt_box):
    """
    Compute overall and per-component NHD.
    """
    components = ['center','dims','R']
    # Get full vertices
    pv = box3d_to_vertices(pred_box['center'], pred_box['dims'], pred_box['R'])
    gv = box3d_to_vertices(gt_box['center'],   gt_box['dims'],   gt_box['R'])
    results = {}
    results['overall'] = calculate_nhd(pv, gv)
    # Disentangle each component by replacing others with ground truth
    for comp in components:
        mod = pred_box.copy()
        for other in components:
            if other != comp:
                mod[other] = gt_box[other]
        mv = box3d_to_vertices(mod['center'], mod['dims'], mod['R'])
        results[comp] = calculate_nhd(mv, gv)
    return results
