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

from custom_trainer import CustomTrainer
from utils.utils import get_2d_coord_np, crop_resize_by_warp_affine, draw_3d_bbox_with_coordinate_frame, draw_3d_bbox_on_image_array, box3d_overlap
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


def main():
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomTrainer(args=args)
    model = model.load_from_checkpoint("logs/objectron-decode_rt/epoch=129.ckpt", args=args)
    model = model.to(device)
    model.eval()
    
    with open("image_cat_id_alignment/objectron_image_path_to_id.json", "r") as f:
        img_to_id = json.load(f)

    with open("image_cat_id_alignment/objectron_category_id_to_name.json", "r") as f:
        category_to_id = json.load(f)
    
    data_path = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets/Omni3D/Objectron_test.json"
    images_folder = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets"
    masks_folder = "/workspace/di38wiq/projects/Anything6D/OV3DD_ablation/datasets/objectron/test_mask"

    with open(data_path, "r") as f:
        data = json.load(f)

    image_ids = [image['id'] for image in data['images']]
    img_info = {image['id']: image for image in data['images']}
    
    annotations = {}
    for annotation in data['annotations']:
        if annotation['image_id'] not in annotations:
            annotations[annotation['image_id']] = [annotation]
        else:
            annotations[annotation['image_id']].append(annotation)

    n = 0

    predictions = []
    for img_id, value in tqdm(img_info.items(), desc="Processing images"):
        img_path = os.path.join(images_folder, value['file_path'])
        if "bike" in img_path:
            continue
        if img_id not in annotations:
            continue
        
        image = Image.open(img_path)
        image = np.array(image.convert("RGB")).astype(np.float32)
        
        image_raw = Image.open(img_path)
        image_raw = np.array(image_raw.convert("RGB"))

        cam_K = np.array(value['K'])

        prediction = {
            "image_id": img_id,
            "K": cam_K.tolist(),
            "width": value["width"],
            "height": value["height"],
            "instances": []
        }
        
        for annotation in annotations[img_id]:
            m2c_R = np.array(annotation['R_cam'])

            center_cam = np.array(annotation['center_cam'])
            bbox_cam = np.array(annotation['bbox3D_cam'])

            bbox_cam_on_img = np.concatenate([center_cam[None], bbox_cam], axis=0) @ cam_K.T
            bbox_cam_on_img = bbox_cam_on_img[:, 0:2] / bbox_cam_on_img[:, 2:]
            bbox = np.array([
                np.min(bbox_cam_on_img[:, 0]),
                np.min(bbox_cam_on_img[:, 1]),
                np.max(bbox_cam_on_img[:, 0]) - np.min(bbox_cam_on_img[:, 0]),
                np.max(bbox_cam_on_img[:, 1]) -np.min(bbox_cam_on_img[:, 1])
            ]).astype(np.uint16)
            
            c = np.array((bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3])) # [c_w, c_h]
            s = max(bbox[2], bbox[3])
            
            mask_path = os.path.join(masks_folder, f"{value['file_path'].split('/')[-1].replace('.jpg', '')}_{img_id}_anno{annotations[img_id].index(annotation)}.png")
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = mask.astype(np.float32) / 255.
            
            image[mask == 0] = 0

            # Image.fromarray(image.astype(np.uint8)).save("test_pose/image_masked.png")
            # Image.fromarray((mask*255.).astype(np.uint8)).save("test_pose/mask.png")
            
            img_roi, c_h_, c_w_, s_, roi_coord_2d = zoom_in(image, c, s, res=args.scale_size, return_roi_coord=True)
            mask_roi, *_ = zoom_in(mask, c, s, res=args.scale_size, interpolate=cv2.INTER_NEAREST)

            # Image.fromarray(img_roi.astype(np.uint8)).save(f"test_pose/image_roi_{n}.png")
            # Image.fromarray((mask_roi*255.).astype(np.uint8)).save(f"test_pose/mask_roi_{n}.png")

            c = np.array([c_w_, c_h_])
            s = s_
            
            with torch.no_grad():
                input_rgb = torch.tensor(img_roi/255.).permute((2, 0, 1)).unsqueeze(0).to(torch.float32).to(device)
                input_mask = torch.tensor(mask_roi).unsqueeze(0).to(torch.float32).to(device)
                input_roi_coord_2d = torch.tensor(roi_coord_2d).unsqueeze(0).to(torch.float32).to(device)
                
                preds = model.model(input_rgb, input_mask, input_roi_coord_2d)
                pred_r, pred_t, pred_dims = preds['pred_r'], preds['pred_t'], preds['pred_dims']
                pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                    pred_r,
                    pred_centroids=pred_t[:, :2],
                    pred_z_vals=pred_t[:, 2:3], # must be [B, 1]
                    roi_cams=torch.tensor(cam_K).unsqueeze(0).to(torch.float32).to(device),
                    roi_centers=torch.tensor(c).unsqueeze(0).to(torch.float32).to(device),
                    resize_ratios=torch.tensor([args.scale_size / s]).to(torch.float32).to(device),
                    roi_whs=torch.tensor([s]).to(torch.float32).to(device),
                    eps=1e-8,
                    is_allo=True,
                    z_type='REL'
                )
                
                kpts_m = torch.zeros([1, 8, 3]).to(torch.float32).to(device)
                l, w, h = pred_dims[:, 0].unsqueeze(1), pred_dims[:, 1].unsqueeze(1), pred_dims[:, 2].unsqueeze(1)
                kpts_m[:, [0, 1, 2, 3], 0], kpts_m[:, [4, 5, 6, 7], 0] = -l/2, l/2
                kpts_m[:, [0, 1, 4, 5], 1], kpts_m[:, [2, 3, 6, 7], 1] = -w/2, w/2
                kpts_m[:, [0, 2, 4, 6], 2], kpts_m[:, [1, 3, 5, 7], 2] = -h/2, h/2
                kpts_pred = transform_pts_batch(kpts_m, pred_ego_rot, pred_trans)
                
                # kps_3d = np.zeros([1, 9, 3]).astype(np.float32)
                # l, w, h = pred_dims[:, 0].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 1].unsqueeze(1).detach().cpu().numpy(), pred_dims[:, 2].unsqueeze(1).detach().cpu().numpy()
                # kps_3d[:, [1, 2, 3, 4], 0], kps_3d[:, [5, 6, 7, 8], 0] = -l/2, l/2
                # kps_3d[:, [1, 2, 5, 6], 1], kps_3d[:, [3, 4, 7, 8], 1] = -w/2, w/2
                # kps_3d[:, [1, 3, 5, 7], 2], kps_3d[:, [2, 4, 6, 8], 2] = -h/2, h/2
                # img_with_pose = draw_3d_bbox_on_image_array(image_raw, bbox_cam_on_img[1:])
                # img_with_pose_pred = draw_3d_bbox_with_coordinate_frame(image_raw, kps_3d[0], pred_ego_rot[0].cpu().numpy(), pred_trans[0].cpu().numpy(), cam_K)
                # Image.fromarray(img_with_pose).save(f"test_pose/gt_pose_{n}.png")
                # Image.fromarray(img_with_pose_pred).save(f"test_pose/pred_pose_{n}.png")
                # print("save pose images")

                n += 1
                

                prediction["instances"].append(
                    {
                        "image_id": img_id,
                        "category_id": annotation["category_id"],
                        "bbox": bbox.tolist(),
                        "score": 1.0,
                        "depth": pred_trans[0].cpu().numpy().tolist()[2],
                        "bbox3D": [
                            kpts_pred[0][0].cpu().numpy().tolist(),
                            kpts_pred[0][4].cpu().numpy().tolist(),
                            kpts_pred[0][6].cpu().numpy().tolist(),
                            kpts_pred[0][2].cpu().numpy().tolist(),
                            kpts_pred[0][1].cpu().numpy().tolist(),
                            kpts_pred[0][5].cpu().numpy().tolist(),
                            kpts_pred[0][7].cpu().numpy().tolist(),
                            kpts_pred[0][3].cpu().numpy().tolist()
                        ],
                        "center_cam": pred_trans[0].cpu().numpy().tolist(),
                        "center_2D": c.tolist(),
                        "dimensions": pred_dims[0].cpu().numpy().tolist(),
                        "pose": pred_ego_rot[0].cpu().numpy().tolist()
                    }
                )

        predictions.append(prediction)
        
    torch.save(predictions, "logs/inference_pth/instances_predictions.pth")



    # # Dataset setting
    # dataset_kwargs = {
    #     'dataset_name': args.dataset, 
    #     'data_path': args.data_path, 
    #     'data_name': args.data_name, 
    #     'data_type': args.data_val,
    #     'feat_3d_path': args.data_3d_feat ,
    #     'xyz_bin': args.nocs_bin,
    #     'scale_size': args.scale_size
    # }
    # dataset_kwargs['scale_size'] = args.scale_size
    # dataset_kwargs['data_type'] = args.data_val

    # inference_dataset = get_dataset(**dataset_kwargs, is_train=False)
    # sampler_infer = RandomSampler(inference_dataset)
    # inference_loader = DataLoader(
    #     inference_dataset,
    #     batch_size=16,
    #     sampler=sampler_infer,
    #     num_workers=args.workers
    # )
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = CustomTrainer(args=args)
    # model = model.load_from_checkpoint("logs/objectron-decode_rt/epoch=129.ckpt", args=args)
    # model = model.to(device)
    # model.eval()

    # with open("image_cat_id_alignment/objectron_image_path_to_id.json", "r") as f:
    #     img_to_id = json.load(f)

    # with open("image_cat_id_alignment/objectron_category_id_to_name.json", "r") as f:
    #     category_to_id = json.load(f)

    # predictions_list = []
    # predictions = {}

    # ious = []
    
    # with torch.no_grad():
    #     for idx, batch in enumerate(tqdm(inference_loader, desc="Inference")):
    #         for key, value in batch.items():
    #             batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    #         loss = model(batch)
    #         output_dict = model.model_output

    #         bs = len(batch['img_name'])
    #         kps_3d = np.zeros([bs, 9, 3]).astype(np.float32)
    #         l, w, h = output_dict["pred_dims"][:, 0].unsqueeze(1).detach().cpu().numpy(), output_dict["pred_dims"][:, 1].unsqueeze(1).detach().cpu().numpy(), output_dict["pred_dims"][:, 2].unsqueeze(1).detach().cpu().numpy()
    #         kps_3d[:, [1, 2, 3, 4], 0], kps_3d[:, [5, 6, 7, 8], 0] = -l/2, l/2
    #         kps_3d[:, [1, 2, 5, 6], 1], kps_3d[:, [3, 4, 7, 8], 1] = -w/2, w/2
    #         kps_3d[:, [1, 3, 5, 7], 2], kps_3d[:, [2, 4, 6, 8], 2] = -h/2, h/2
            
    #         for i in range(len(batch['img_name'])):
    #             img_name = batch['img_name'][i]
    #             category_name = batch['class_name'][i]
    #             if img_name in img_to_id and category_name in category_to_id:
    #                 img_id = img_to_id[img_name]
    #                 category_id = category_to_id[category_name]
    #                 if str(img_id) not in predictions:
    #                     predictions[str(img_id)] = {
    #                         "image_id": img_id,
    #                         "K": batch["cam"][i].cpu().numpy().tolist(),
    #                         "width": int(batch["raw_scene"][i].shape[2]),
    #                         "height": int(batch["raw_scene"][i].shape[1]),
    #                         "instances": []
    #                     }
    #                     predictions[str(img_id)]["instances"].append(
    #                         {
    #                             "image_id": img_id,
    #                             "category_id": category_id,
    #                             "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
    #                             "score": 1.0,
    #                             "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
    #                             "bbox3D": [
    #                                 output_dict["pred_pts_3d"][i][0].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][4].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][6].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][2].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][1].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][5].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][7].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][3].cpu().numpy().tolist()
    #                             ],
    #                             "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
    #                             "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
    #                             "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
    #                             "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
    #                         }
    #                     )
    #                 else:
    #                     predictions[str(img_id)]["instances"].append(
    #                         {
    #                             "image_id": img_id,
    #                             "category_id": category_id,
    #                             "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
    #                             "score": 1.0,
    #                             "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
    #                             "bbox3D": [
    #                                 output_dict["pred_pts_3d"][i][0].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][4].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][6].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][2].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][1].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][5].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][7].cpu().numpy().tolist(),
    #                                 output_dict["pred_pts_3d"][i][3].cpu().numpy().tolist()
    #                             ],
    #                             "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
    #                             "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
    #                             "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
    #                             "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
    #                         }
    #                     )

    #                 # pred = torch.stack(
    #                 #     [
    #                 #         output_dict["pred_pts_3d"][i][0],
    #                 #         output_dict["pred_pts_3d"][i][4],
    #                 #         output_dict["pred_pts_3d"][i][6],
    #                 #         output_dict["pred_pts_3d"][i][2],
    #                 #         output_dict["pred_pts_3d"][i][1],
    #                 #         output_dict["pred_pts_3d"][i][5],
    #                 #         output_dict["pred_pts_3d"][i][7],
    #                 #         output_dict["pred_pts_3d"][i][3]
    #                 #     ], dim=0
    #                 # ).unsqueeze(0).to(output_dict["pred_pts_3d"])

    #                 # gt = torch.stack(
    #                 #     [
    #                 #         output_dict["gt_pts_3d"][i][0],
    #                 #         output_dict["gt_pts_3d"][i][4],
    #                 #         output_dict["gt_pts_3d"][i][6],
    #                 #         output_dict["gt_pts_3d"][i][2],
    #                 #         output_dict["gt_pts_3d"][i][1],
    #                 #         output_dict["gt_pts_3d"][i][5],
    #                 #         output_dict["gt_pts_3d"][i][7],
    #                 #         output_dict["gt_pts_3d"][i][3]
    #                 #     ], dim=0
    #                 # ).unsqueeze(0).to(output_dict["gt_pts_3d"])

    #                 # iou = box3d_overlap(output_dict["pred_pts_3d"][i].unsqueeze(0), output_dict["gt_pts_3d"][i].unsqueeze(0))

    #             else:
    #                 continue

    # for key, value in predictions.items():
    #     predictions_list.append(value)

    # torch.save(predictions_list, "logs/inference_pth/instances_predictions_rt_objectron.pth")


if __name__ == '__main__':
    main()