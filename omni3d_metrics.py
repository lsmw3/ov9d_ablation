"""
evaluate_3d.py

Compute 3D metrics per category and overall, **without** COCO:
  • AP   : mean AP over IoU thresholds [0.05,0.10,…,0.50]
  • AP15 : AP at IoU=0.15
  • AP25 : AP at IoU=0.25
  • AP50 : AP at IoU=0.50
  • APn  : AP for “near” objects    (depth∈[0,10])
  • APm  : AP for “medium” objects  (depth∈(10,35])
  • APf  : AP for “far” objects     (depth>35)
  • AR1, AR10, AR100 : average recall at max detections 1,10,100

Your GT/Pred JSONs must each be a list of dicts:
  { "image_id": …,
    "category_name": "chair",
    "bbox3D": [ [x,y,z],…x8 ],
    "score": …?        # only for preds
  }
"""
import argparse, json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# --- minimal IoU3D from OMNI3D ---
_BOX_PLANES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
_BOX_TRIANGLES = [[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                  [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                  [1,2,6],[1,6,5],[0,3,7],[0,7,4]]

def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that each of the 6 faces of each box is coplanar.
    boxes: (B, 8, 3) tensor of box corners
    Returns a BoolTensor of shape (B,) indicating which boxes are valid.
    """
    # faces: (6 faces × 4 corners)
    faces = torch.tensor(_BOX_PLANES, dtype=torch.int64, device=boxes.device)  # (6,4)
    B = boxes.shape[0]
    P, V = faces.shape  # P=6, V=4

    # gather the 4 vertices for each face: result (B, P, V, 3)
    verts = boxes[:, faces]  

    # split into the first three (v0,v1,v2) and the fourth (v3)
    v0, v1, v2, v3 = verts.unbind(dim=2)  # each is (B, P, 3)

    # face normal from first three verts
    e0 = F.normalize(v1 - v0, dim=-1)  # (B, P, 3)
    e1 = F.normalize(v2 - v0, dim=-1)  # (B, P, 3)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)  # (B, P, 3)

    # vector from v0 to the fourth vertex
    offset = v3 - v0  # (B, P, 3)

    # dot product along last dim
    d = (offset * normal).sum(dim=-1)  # (B, P)

    # must be near zero for all faces
    return (d.abs() < eps).all(dim=1)

def _check_nonzero(boxes, eps=1e-8):
    faces = torch.tensor(_BOX_TRIANGLES, dtype=torch.int64, device=boxes.device)
    B,_,_ = boxes.shape; T,V = faces.shape
    verts = boxes.index_select(1, faces.reshape(-1))
    v0,v1,v2 = verts.reshape(B,T,V,3).unbind(2)
    normals = torch.cross(v1-v0, v2-v0, dim=-1)
    areas = normals.norm(dim=-1)/2
    return (areas>eps).all(dim=1)

def box3d_overlap(dt, gt, eps_coplanar=1e-4, eps_nonzero=1e-8):
    invalid = ~_check_coplanar(dt) | ~_check_nonzero(dt)
    import pytorch3d._C as _C
    ious = _C.iou_box3d(dt,gt)[1]
    if invalid.any():
        ious[invalid] = 0.0
    return ious

def compute_ious3d(preds, gts):
    if len(preds)==0 or len(gts)==0:
        return np.zeros((len(preds),len(gts)),dtype=float)
    dt = torch.tensor(preds,dtype=torch.float32)
    gt = torch.tensor(gts,  dtype=torch.float32)
    with torch.no_grad():
        M = box3d_overlap(dt,gt)
    return M.cpu().numpy()

# --- VOC‐style AP integration ---
def voc_ap(rec,prec):
    mrec = np.concatenate(([0.],rec,[1.]))
    mpre = np.concatenate(([0.],prec,[0.]))
    for i in range(len(mpre)-1,0,-1):
        mpre[i-1] = max(mpre[i-1],mpre[i])
    idx = np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))

def evaluate_category(preds, gts, iou_thrs, max_det):
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    pb = [p['bbox3D'] for p in preds]
    gb = [g['bbox3D'] for g in gts]
    ious = compute_ious3d(pb,gb)
    n_gt = len(gb)
    ap_thr = []; rec_thr = []
    for thr in iou_thrs:
        tp = np.zeros(len(preds),int); fp = np.zeros(len(preds),int)
        matched = set()
        for i,p in enumerate(preds[:max_det]):
            if n_gt==0:
                fp[i]=1; continue
            row = ious[i]
            j = int(np.argmax(row))
            if row[j]>=thr and j not in matched:
                tp[i]=1; matched.add(j)
            else:
                fp[i]=1
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c/(n_gt+1e-8); prec=tp_c/(tp_c+fp_c+1e-8)
        ap_thr.append(voc_ap(rec,prec))
        rec_thr.append(rec[-1] if len(rec)>0 else 0.)
    return float(np.mean(ap_thr)), np.array(rec_thr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   required=True,
                        help="GT JSON list of {image_id,category_name,bbox3D}")
    parser.add_argument("--pred", required=True,
                        help="Pred JSON list of {image_id,category_name,bbox3D,score}")
    args = parser.parse_args()

    gt_list   = json.load(open(args.gt))
    pred_list = json.load(open(args.pred))

    # IoU thresholds
    all_iou_thrs = np.arange(0.05,0.51,0.05)
    fixed = {'AP15':0.15, 'AP25':0.25, 'AP50':0.50}

    # depth ranges
    depth_ranges = {
      'APn': (0.0,10.0),
      'APm': (10.0,35.0),
      'APf': (35.0,1e8),
    }

    # group by category
    gt_by_cat   = defaultdict(list)
    pd_by_cat   = defaultdict(list)
    for g in gt_list:   gt_by_cat[g['category_name']].append(g)
    for p in pred_list: pd_by_cat[p['category_name']].append(p)

    metrics_by_cat = {}
    for cat in sorted(set(gt_by_cat)|set(pd_by_cat)):
        gts  = gt_by_cat.get(cat, [])
        preds= pd_by_cat.get(cat, [])

        # 1) AP = mean over all IoU thresholds
        AP, _ = evaluate_category(preds, gts, all_iou_thrs, max_det=100)

        # 2) AP15,25,50
        AP15,_ = evaluate_category(preds, gts, np.array([fixed['AP15']]), max_det=100)
        AP25,_ = evaluate_category(preds, gts, np.array([fixed['AP25']]), max_det=100)
        AP50,_ = evaluate_category(preds, gts, np.array([fixed['AP50']]), max_det=100)

        # 3) APn,APm,APf
        APn = APm = APf = 0.0
        for key,(d0,d1) in depth_ranges.items():
            sub_g = [g for g in gts   if d0 <= g['depth'] < d1]
            sub_p = [p for p in preds if d0 <= p['depth'] < d1]
            ap_sub,_ = evaluate_category(sub_p, sub_g, all_iou_thrs, max_det=100)
            if key=='APn': APn=ap_sub
            if key=='APm': APm=ap_sub
            if key=='APf': APf=ap_sub

        # 4) AR1, AR10, AR100
        _, rec1  = evaluate_category(preds, gts, all_iou_thrs, max_det=1)
        _, rec10 = evaluate_category(preds, gts, all_iou_thrs, max_det=10)
        _, rec100= evaluate_category(preds, gts, all_iou_thrs, max_det=100)
        AR1   = float(np.mean(rec1))
        AR10  = float(np.mean(rec10))
        AR100 = float(np.mean(rec100))

        metrics_by_cat[cat] = {
            'AP':   AP*100,
            'AP15': AP15*100,
            'AP25': AP25*100,
            'AP50': AP50*100,
            'APn':  APn*100,
            'APm':  APm*100,
            'APf':  APf*100,
            'AR1':  AR1*100,
            'AR10': AR10*100,
            'AR100':AR100*100,
        }

    # print per‐category
    headers = ["AP","AP15","AP25","AP50","APn","APm","APf","AR1","AR10","AR100"]
    print(f"{'Category':15s} " + " ".join(f"{h:7s}" for h in headers))
    for cat,m in metrics_by_cat.items():
        vals = [m[h] for h in headers]
        print(f"{cat:15s} " + " ".join(f"{v:7.2f}" for v in vals))

    # overall mean
    print("\nOverall means:")
    for h in headers:
        arr = [m[h] for m in metrics_by_cat.values()]
        print(f"  {h:6s} = {np.mean(arr):6.2f}")