import os
import numpy as np
import cv2
import json
from datetime import datetime
import pickle
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from custom_trainer_3 import CustomTrainer
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, draw_3d_bbox_with_coordinate_frame, draw_3d_bbox_on_image_array, box3d_overlap, draw_bbox_on_image_array
from utils.pose_postprocessing import pose_from_pred_centroid_z

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions


def zoom_in(im, c, s, res, interpolate=cv2.INTER_LINEAR, return_roi_coord=False):
    c_w, c_h = c
    c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
    ndim = im.ndim
    if ndim == 2:
        im = im[..., np.newaxis]

    max_h, max_w = im.shape[0:2]
    s = s

    im_crop = crop_resize_by_warp_affine(im, np.array([c_w, c_h]), s, res).astype(np.float32)

    if return_roi_coord:
        coord_2d = get_2d_coord_np(max_w, max_h, fmt="HWC")
        roi_coord_2d = crop_resize_by_warp_affine(coord_2d, np.array([c_w, c_h]), s, res, interpolation=interpolate).transpose(2, 0, 1).astype(np.float32) # HWC -> CHW
        return im_crop, c_h, c_w, s, roi_coord_2d

    return im_crop, c_h, c_w, s


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


def preprocess(image, bbox, K, mask, target_size=(490, 490)):
    """
    Pad image to square and resize while adjusting bounding boxes
    
    Args:
        image: numpy array of shape (H, W, 3)
        bboxes: list of [x, y, w, h] coordinates in original image
        K: initial camera instrinsic
        mask: numpy array of shape (H, W)
        nocs: numpy array of shape (H, W, 3)
        target_size: desired output size (height, width)
    
    Returns:
        padded_resized_img: numpy array of padded and resized image
        adjusted_bboxes: list of adjusted bounding boxes
    """
    H, W, _ = image.shape
    max_dim = max(H, W)
    
    # Create a square black canvas
    padded_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded_mask = np.zeros((max_dim, max_dim), dtype=mask.dtype)
    
    # Paste original image onto canvas (centered)
    pad_top = (max_dim - H) // 2
    pad_left = (max_dim - W) // 2

    padded_img[pad_top:pad_top+H, pad_left:pad_left+W, :] = image
    padded_mask[pad_top:pad_top+H, pad_left:pad_left+W] = mask

    K_pad = K.copy()
    K_pad[0,2] += pad_left
    K_pad[1,2] += pad_top
    
    # Calculate scale factor for resizing
    scale_factor = target_size[0] / max_dim
    
    # Adjust bboxes for padding and resizing
    x, y, w, h = bbox
    
    # Add padding offsets to top-left coordinates
    x_padded = x + pad_left
    y_padded = y + pad_top
    
    # Width and height remain the same after padding
    w_padded = w
    h_padded = h
    
    # Apply scaling for resize
    x_adj = x_padded * scale_factor
    y_adj = y_padded * scale_factor
    w_adj = w_padded * scale_factor
    h_adj = h_padded * scale_factor
        
    adjusted_bboxes = np.array([x_adj, y_adj, w_adj, h_adj])

    K_resized = K_pad.copy()
    K_resized[0,:] *= scale_factor
    K_resized[1,:] *= scale_factor
    
    # Resize images
    S_h, S_w = target_size
    padded_resized_img = cv2.resize(padded_img, (S_w, S_h), interpolation=cv2.INTER_LINEAR)
    padded_resized_mask = cv2.resize(padded_mask, (S_w, S_h), interpolation=cv2.INTER_NEAREST)

    
    return padded_resized_img, adjusted_bboxes, K_resized, padded_resized_mask, scale_factor


def preprocess_no_pad(image, bbox, K, mask, target_size=(480, 360)):
    """
    Resize image directly to target size without padding
    
    Args:
        image: numpy array of shape (H, W, 3)
        bbox: array of [x, y, w, h] coordinates in original image
        K: initial camera intrinsic matrix
        mask: numpy array of shape (H, W)
        target_size: desired output size (height, width)
    
    Returns:
        resized_img: numpy array of resized image
        adjusted_bbox: adjusted bounding box coordinates
        K_resized: adjusted camera intrinsic matrix
        resized_mask: resized mask
        scale_factors: tuple containing (width_scale, height_scale)
    """
    H, W, _ = image.shape
    
    # Calculate separate scale factors for height and width
    height_scale = target_size[0] / H
    width_scale = target_size[1] / W
    
    # Adjust bounding box for direct resizing
    x, y, w, h = bbox
    
    # Apply separate scaling to coordinates and dimensions
    x_adj = x * width_scale
    y_adj = y * height_scale
    w_adj = w * width_scale
    h_adj = h * height_scale
    
    adjusted_bbox = np.array([x_adj, y_adj, w_adj, h_adj])
    
    # Adjust camera intrinsic matrix for non-uniform scaling
    K_resized = K.copy()
    # Scale fx and cx (width-related parameters)
    K_resized[0, 0] *= width_scale  # fx
    K_resized[0, 2] *= width_scale  # cx (principal point x)
    # Scale fy and cy (height-related parameters)
    K_resized[1, 1] *= height_scale  # fy
    K_resized[1, 2] *= height_scale  # cy (principal point y)
    
    # Resize image and mask directly to target size
    # Note: cv2.resize takes (width, height) order
    resized_img = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Return scale factors as a tuple for potential later use
    scale_factors = (width_scale, height_scale)
    
    return resized_img, adjusted_bbox, K_resized, resized_mask, scale_factors


def main():
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)


    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CustomTrainer(args=args)
    # model = model.load_from_checkpoint("logs/objectron_3-decode_rt/epoch=84.ckpt", args=args)
    # model = model.to(device)
    # model.eval()
    
    # with open("image_cat_id_alignment/objectron_image_path_to_id.json", "r") as f:
    #     img_to_id = json.load(f)

    # with open("image_cat_id_alignment/objectron_category_id_to_name.json", "r") as f:
    #     category_to_id = json.load(f)
    
    # data_path = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets/Omni3D/Objectron_test.json"
    # images_folder = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets"
    # masks_folder = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets/objectron/test_mask"

    # with open(data_path, "r") as f:
    #     data = json.load(f)

    # image_ids = [image['id'] for image in data['images']]
    # img_info = {image['id']: image for image in data['images']}
    
    # annotations = {}
    # for annotation in data['annotations']:
    #     if annotation['image_id'] not in annotations:
    #         annotations[annotation['image_id']] = [annotation]
    #     else:
    #         annotations[annotation['image_id']].append(annotation)

    # # n = 0

    # predictions = []
    # for img_id, value in tqdm(img_info.items(), desc="Processing images"):
    #     img_path = os.path.join(images_folder, value['file_path'])
    #     if "bike" in img_path:
    #         continue
    #     if img_id not in annotations:
    #         continue
        
    #     image = Image.open(img_path)
    #     image = np.array(image.convert("RGB")).astype(np.float32)
        
    #     image_raw = Image.open(img_path)
    #     image_raw = np.array(image_raw.convert("RGB"))

    #     cam_K = np.array(value['K'])

    #     prediction = {
    #         "image_id": img_id,
    #         "K": cam_K.tolist(),
    #         "width": value["width"],
    #         "height": value["height"],
    #         "instances": []
    #     }
        
    #     for annotation in annotations[img_id]:
    #         m2c_R = np.array(annotation['R_cam'])

    #         center_cam = np.array(annotation['center_cam'])
    #         bbox_cam = np.array(annotation['bbox3D_cam'])

    #         bbox_with_center = np.concatenate([center_cam[None], bbox_cam], axis=0)
    #         bbox_m = (bbox_with_center - center_cam[None]) @ m2c_R

    #         bbox_cam_on_img = np.concatenate([center_cam[None], bbox_cam], axis=0) @ cam_K.T
    #         bbox_cam_on_img = bbox_cam_on_img[:, 0:2] / bbox_cam_on_img[:, 2:]
    #         bbox = np.array([
    #             np.min(bbox_cam_on_img[:, 0]),
    #             np.min(bbox_cam_on_img[:, 1]),
    #             np.max(bbox_cam_on_img[:, 0]) - np.min(bbox_cam_on_img[:, 0]),
    #             np.max(bbox_cam_on_img[:, 1]) -np.min(bbox_cam_on_img[:, 1])
    #         ]).astype(np.uint16)
            
    #         mask_path = os.path.join(masks_folder, f"{value['file_path'].split('/')[-1].replace('.jpg', '')}_{img_id}_anno{annotations[img_id].index(annotation)}.png")
    #         mask = Image.open(mask_path)
    #         mask = np.array(mask)
    #         mask = mask.astype(np.float32) / 255.

    #         scene_resized_ori, bbox_resized_ori, cam_K_resized_ori, mask_resized_ori, scale_ratio_ori = preprocess_no_pad(image_raw, bbox, cam_K, mask, target_size=(480, 360))

    #         # ########################################
    #         # if not os.path.exists(f"test_pose/sub_test_9k/{n}"):
    #         #     os.mkdir(f"test_pose/sub_test_9k/{n}")
            
    #         # print(scene_resized_ori.shape)
            
    #         # Image.fromarray(image.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image.png")
    #         # Image.fromarray((mask*255.).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/mask.png")
    #         # Image.fromarray((mask[..., None]*image).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_masked.png")
    #         # Image.fromarray(scene_resized_ori.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_resized_ori.png")
    #         # Image.fromarray((mask_resized_ori*255.).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/mask_resized_ori.png")
    #         # Image.fromarray((mask_resized_ori[..., None]*scene_resized_ori).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_masked_resized_ori.png")

    #         # img_with_2d_bbox = draw_bbox_on_image_array(image.copy(), bbox)
    #         # Image.fromarray(img_with_2d_bbox.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_with_2d_bbox.png")
    #         # img_resized_with_2d_bbox = draw_bbox_on_image_array(scene_resized_ori.copy(), bbox_resized_ori)
    #         # Image.fromarray(img_resized_with_2d_bbox.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_resized_with_2d_bbox.png")
    #         # ########################################
            
    #         c_raw = np.array((bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3])) # [c_w, c_h]
    #         s_raw = max(bbox[2], bbox[3])

    #         c = np.array((bbox_resized_ori[0]+0.5*bbox_resized_ori[2], bbox_resized_ori[1]+0.5*bbox_resized_ori[3])) # [c_w, c_h]
    #         s = max(bbox_resized_ori[2], bbox_resized_ori[3])

    #         img_roi, c_h_, c_w_, s_, roi_coord_2d = zoom_in(scene_resized_ori, c, s, res=args.scale_size, return_roi_coord=True)
    #         mask_roi, *_ = zoom_in(mask_resized_ori, c, s, res=args.scale_size, interpolate=cv2.INTER_NEAREST)

    #         scene_padded_resized, bbox_resized, cam_K_resized, mask_resized, scale_ratio = preprocess(scene_resized_ori, bbox_resized_ori, cam_K_resized_ori, mask_resized_ori)

    #         c_resized = np.array((bbox_resized[0]+0.5*bbox_resized[2], bbox_resized[1]+0.5*bbox_resized[3])) # [c_w, c_h]
    #         s_resized = max(bbox_resized[2], bbox_resized[3])

    #         img_latent, *_ = zoom_in(scene_padded_resized, c_resized, s_resized, res=args.latent_size)
    #         mask_latent, *_ = zoom_in(mask_resized, c_resized, s_resized, res=args.latent_size, interpolate=cv2.INTER_NEAREST)

    #         img_roi_latent = cv2.resize(img_latent, (args.scale_size, args.scale_size), interpolation=cv2.INTER_LINEAR)

    #         # #############################################################
    #         # Image.fromarray(img_roi.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_roi.png")
    #         # Image.fromarray((mask_roi*255.).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/mask_roi.png")
    #         # Image.fromarray((mask_roi[..., None]*img_roi).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_masked_roi.png")

    #         # Image.fromarray(scene_padded_resized.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/input_scene.png")
    #         # Image.fromarray((mask_resized*255.).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/mask_resized.png")
    #         # Image.fromarray((mask_resized[..., None]*scene_padded_resized).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/input_scene_masked.png")

    #         # img_with_2d_bbox_resized = draw_bbox_on_image_array(scene_padded_resized.copy(), bbox_resized)
    #         # Image.fromarray(img_with_2d_bbox_resized.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/input_scene_with_2d_bbox.png")

    #         # Image.fromarray(img_latent.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_latent.png")
    #         # Image.fromarray((mask_latent*255.).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/mask_latent.png")
    #         # Image.fromarray((mask_latent[..., None]*img_latent).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_masked_latent.png")

    #         # Image.fromarray(img_roi_latent.astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_roi_latent.png")
    #         # Image.fromarray((mask_roi[..., None]*img_roi_latent).astype(np.uint8)).save(f"test_pose/sub_test_9k/{n}/image_masked_roi_latent.png")
    #         # #############################################################

    #         c = np.array([c_w_, c_h_])
    #         s = s_

    #         with torch.no_grad():
    #             input_rgb = torch.tensor(scene_padded_resized/255.).permute((2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
    #             input_mask = torch.tensor(mask_roi).unsqueeze(0).to(torch.float32).to(device)
    #             input_mask_latent = torch.tensor(mask_latent).unsqueeze(0).to(torch.float32).to(device)
    #             input_c = torch.tensor(c_resized).unsqueeze(0).to(torch.float32).to(device)
    #             input_s = torch.tensor([s_resized]).to(torch.float32).to(device)
    #             input_roi_coord_2d = torch.tensor(roi_coord_2d).unsqueeze(0).to(torch.float32).to(device)
                
    #             preds = model.model(input_rgb, input_mask, input_mask_latent, input_c, input_s, input_roi_coord_2d)
    #             pred_r, pred_t, pred_dims = preds['pred_r'], preds['pred_t'], preds['pred_dims']
    #             pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
    #                 pred_r,
    #                 pred_centroids=pred_t[:, :2],
    #                 pred_z_vals=pred_t[:, 2:3], # must be [B, 1]
    #                 roi_cams=torch.tensor(cam_K_resized_ori).unsqueeze(0).to(torch.float32).to(device),
    #                 roi_centers=torch.tensor(c).unsqueeze(0).to(torch.float32).to(device),
    #                 resize_ratios=torch.tensor([args.scale_size / s]).to(torch.float32).to(device),
    #                 roi_whs=torch.tensor([s]).to(torch.float32).to(device),
    #                 eps=1e-8,
    #                 is_allo=True,
    #                 z_type='REL'
    #             )
                
    #             kpts_m = torch.zeros([1, 8, 3]).to(torch.float32).to(device)
    #             l, w, h = pred_dims[:, 0].unsqueeze(1), pred_dims[:, 1].unsqueeze(1), pred_dims[:, 2].unsqueeze(1)
    #             kpts_m[:, [0, 1, 2, 3], 0], kpts_m[:, [4, 5, 6, 7], 0] = -l/2, l/2
    #             kpts_m[:, [0, 1, 4, 5], 1], kpts_m[:, [2, 3, 6, 7], 1] = -w/2, w/2
    #             kpts_m[:, [0, 2, 4, 6], 2], kpts_m[:, [1, 3, 5, 7], 2] = -h/2, h/2
    #             kpts_pred = transform_pts_batch(kpts_m, pred_ego_rot, pred_trans)
                
    #             # ########################################################
    #             # kps_3d = np.zeros([1, 9, 3]).astype(np.float32)
    #             # l, w, h = pred_dims[:, 0].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 1].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 2].unsqueeze(1).detach().cpu().numpy()
    #             # kps_3d[:, [1, 2, 3, 4], 0], kps_3d[:, [5, 6, 7, 8], 0] = -l/2, l/2
    #             # kps_3d[:, [1, 2, 5, 6], 1], kps_3d[:, [3, 4, 7, 8], 1] = -w/2, w/2
    #             # kps_3d[:, [1, 3, 5, 7], 2], kps_3d[:, [2, 4, 6, 8], 2] = -h/2, h/2
    #             # # img_with_pose = draw_3d_bbox_on_image_array(image_raw, np.concatenate([center_cam[None], bbox_cam], axis=0), cam_K)
    #             # img_with_pose = draw_3d_bbox_with_coordinate_frame(image_raw, bbox_m, m2c_R, center_cam, cam_K)
    #             # img_with_pose_pred = draw_3d_bbox_with_coordinate_frame(image_raw, kps_3d[0], pred_ego_rot[0].cpu().numpy(), pred_trans[0].cpu().numpy(), cam_K)
    #             # Image.fromarray(img_with_pose).save(f"test_pose/sub_test_9k/{n}/gt_pose.png")
    #             # Image.fromarray(img_with_pose_pred).save(f"test_pose/sub_test_9k/{n}/pred_pose.png")
    #             # print("save pose images")
    #             # ########################################################

    #             # if n != 5:
    #             #     n += 1
    #             #     continue
    #             # else:
    #             #     a

    #             prediction["instances"].append(
    #                 {
    #                     "image_id": img_id,
    #                     "category_id": annotation["category_id"],
    #                     "bbox": bbox.tolist(),
    #                     "score": 1.0,
    #                     "depth": pred_trans[0].cpu().numpy().tolist()[2],
    #                     "bbox3D": [
    #                         kpts_pred[0][0].cpu().numpy().tolist(),
    #                         kpts_pred[0][4].cpu().numpy().tolist(),
    #                         kpts_pred[0][6].cpu().numpy().tolist(),
    #                         kpts_pred[0][2].cpu().numpy().tolist(),
    #                         kpts_pred[0][1].cpu().numpy().tolist(),
    #                         kpts_pred[0][5].cpu().numpy().tolist(),
    #                         kpts_pred[0][7].cpu().numpy().tolist(),
    #                         kpts_pred[0][3].cpu().numpy().tolist()
    #                     ],
    #                     "center_cam": pred_trans[0].cpu().numpy().tolist(),
    #                     "center_2D": c_raw.tolist(),
    #                     "dimensions": pred_dims[0].cpu().numpy().tolist(),
    #                     "pose": pred_ego_rot[0].cpu().numpy().tolist()
    #                 }
    #             )

    #     predictions.append(prediction)
        
    # torch.save(predictions, "logs/inference_pth/instances_predictions_rt_objectron_3_9k.pth")



    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset, 
        'data_path': args.data_path, 
        'data_name': args.data_name, 
        'data_type': args.data_val,
        'feat_3d_path': args.data_3d_feat ,
    }
    dataset_kwargs['scale_size'] = args.scale_size
    dataset_kwargs['latent_size'] = args.latent_size
    dataset_kwargs['virtual_focal'] = args.virtual_focal

    inference_dataset = get_dataset(**dataset_kwargs, is_train=False)
    sampler_infer = RandomSampler(inference_dataset)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=16,
        sampler=sampler_infer,
        num_workers=args.workers
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomTrainer(args=args)
    model = model.load_from_checkpoint("logs/objectron_3-decode_rt/epoch=84.ckpt", args=args)
    model = model.to(device)
    model.eval()

    with open("image_cat_id_alignment/objectron_image_path_to_id.json", "r") as f:
        img_to_id = json.load(f)

    with open("image_cat_id_alignment/objectron_category_id_to_name.json", "r") as f:
        category_to_id = json.load(f)

    predictions_list = []
    predictions = {}
    
    predictions_gts_predrt_list = []
    predictions_gts_predrt = {}
    
    predictions_gtrt_preds_list = []
    predictions_gtrt_preds = {}
    
    predictions_gt_list = []
    predictions_gt = {}
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(inference_loader, desc="Inference")):
            for key, value in batch.items():
                batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            loss = model(batch)
            output_dict = model.model_output

            bs = len(batch['img_name'])
            
            gt_kps_3d = batch['kps_3d_m']
            gt_r = batch['gt_r']
            gt_t = batch['gt_t']
            
            pred_kps_3d = torch.zeros([bs, 9, 3]).to(gt_kps_3d)
            l, w, h = output_dict["pred_dims"][:, 0].unsqueeze(1), output_dict["pred_dims"][:, 1].unsqueeze(1), output_dict["pred_dims"][:, 2].unsqueeze(1)
            pred_kps_3d[:, [1, 2, 3, 4], 0], pred_kps_3d[:, [5, 6, 7, 8], 0] = -l/2, l/2
            pred_kps_3d[:, [1, 2, 5, 6], 1], pred_kps_3d[:, [3, 4, 7, 8], 1] = -w/2, w/2
            pred_kps_3d[:, [1, 3, 5, 7], 2], pred_kps_3d[:, [2, 4, 6, 8], 2] = -h/2, h/2
            pred_r = output_dict["pred_ego_rot"]
            pred_t = output_dict["pred_trans"]
            
            pred_pts_gt_size = transform_pts_batch(gt_kps_3d[:, 1:, :], pred_r, pred_t)
            pred_pts_gt_rt = transform_pts_batch(pred_kps_3d[:, 1:, :], gt_r, gt_t)
            # pred_pts_gt_r = transform_pts_batch(pred_kps_3d[:, 1:, :], gt_r, pred_t)
            # pred_pts_gt_t = transform_pts_batch(pred_kps_3d[:, 1:, :], pred_r, gt_t)
            pred_pts = output_dict["pred_pts_3d"]
            gt_pts = transform_pts_batch(gt_kps_3d[:, 1:, :], gt_r, gt_t)
            
            for i in range(bs):
                img_name = batch['img_name'][i]
                category_name = batch['class_name'][i]
                if img_name in img_to_id and category_name in category_to_id:
                    img_id = img_to_id[img_name]
                    category_id = category_to_id[category_name]
                    for pred_dict, pred_pts_3d in [(predictions, pred_pts), (predictions_gts_predrt, pred_pts_gt_size), (predictions_gtrt_preds, pred_pts_gt_rt), (predictions_gt, gt_pts)]:
                        if str(img_id) not in pred_dict:
                            pred_dict[str(img_id)] = {
                                "image_id": img_id,
                                "K": batch["cam_K"][i].cpu().numpy().tolist(),
                                "width": int(batch["raw_scene"][i].shape[2]),
                                "height": int(batch["raw_scene"][i].shape[1]),
                                "instances": []
                            }
                            pred_dict[str(img_id)]["instances"].append(
                                {
                                    "image_id": img_id,
                                    "category_id": category_id,
                                    "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
                                    "score": 1.0,
                                    "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
                                    "bbox3D": [
                                        pred_pts_3d[i][0].cpu().numpy().tolist(),
                                        pred_pts_3d[i][4].cpu().numpy().tolist(),
                                        pred_pts_3d[i][6].cpu().numpy().tolist(),
                                        pred_pts_3d[i][2].cpu().numpy().tolist(),
                                        pred_pts_3d[i][1].cpu().numpy().tolist(),
                                        pred_pts_3d[i][5].cpu().numpy().tolist(),
                                        pred_pts_3d[i][7].cpu().numpy().tolist(),
                                        pred_pts_3d[i][3].cpu().numpy().tolist()
                                    ],
                                    "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
                                    "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
                                    "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
                                    "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
                                }
                            )
                        else:
                            pred_dict[str(img_id)]["instances"].append(
                                {
                                    "image_id": img_id,
                                    "category_id": category_id,
                                    "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
                                    "score": 1.0,
                                    "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
                                    "bbox3D": [
                                        pred_pts_3d[i][0].cpu().numpy().tolist(),
                                        pred_pts_3d[i][4].cpu().numpy().tolist(),
                                        pred_pts_3d[i][6].cpu().numpy().tolist(),
                                        pred_pts_3d[i][2].cpu().numpy().tolist(),
                                        pred_pts_3d[i][1].cpu().numpy().tolist(),
                                        pred_pts_3d[i][5].cpu().numpy().tolist(),
                                        pred_pts_3d[i][7].cpu().numpy().tolist(),
                                        pred_pts_3d[i][3].cpu().numpy().tolist()
                                    ],
                                    "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
                                    "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
                                    "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
                                    "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
                                }
                            )
                else:
                    continue

    for pred_dict, pred_list in [(predictions, predictions_list), (predictions_gts_predrt, predictions_gts_predrt_list), (predictions_gtrt_preds, predictions_gtrt_preds_list), (predictions_gt, predictions_gt_list)]:
        for key, value in pred_dict.items():
            pred_list.append(value)

    torch.save(predictions_list, "logs/inference_ablation/instances_predictions_rt_objectron_pred.pth")
    torch.save(predictions_gts_predrt_list, "logs/inference_ablation/instances_predictions_rt_objectron_gts_predrt.pth")
    torch.save(predictions_gtrt_preds_list, "logs/inference_ablation/instances_predictions_rt_objectron_gtrt_preds.pth")
    torch.save(predictions_gt_list, "logs/inference_ablation/instances_predictions_rt_objectron_gt.pth")


if __name__ == '__main__':
    main()