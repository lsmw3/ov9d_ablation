import os
import numpy as np
import json
from datetime import datetime
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from custom_trainer import CustomTrainer

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions


def main():
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset, 
        'data_path': args.data_path, 
        'data_name': args.data_name, 
        'data_type': args.data_val,
        'feat_3d_path': args.data_3d_feat ,
        'xyz_bin': args.nocs_bin,
        'raw_w': args.raw_w,
        'raw_h': args.raw_h,
        'scale_size': args.scale_size
    }
    dataset_kwargs['scale_size'] = args.scale_size
    dataset_kwargs['data_type'] = args.data_val

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
    model = model.load_from_checkpoint("logs/attn_rt/epoch=104.ckpt", args=args)
    model = model.to(device)
    model.eval()

    with open("objectron_image_path_to_id.json", "r") as f:
        img_to_id = json.load(f)

    with open("objectron_category_id_to_name.json", "r") as f:
        category_to_id = json.load(f)

    predictions_list = []
    predictions = {}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(inference_loader, desc="Inference")):
            for key, value in batch.items():
                batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
            loss = model(batch)
            output_dict = model.model_output
            for i in range(len(batch['img_name'])):
                img_name = batch['img_name'][i]
                category_name = batch['class_name'][i]
                if img_name in img_to_id and category_name in category_to_id:
                    img_id = img_to_id[img_name]
                    category_id = category_to_id[category_name]
                    if str(img_id) not in predictions:
                        predictions[str(img_id)] = {
                            "image_id": img_id,
                            "K": batch["cam"][i].cpu().numpy().tolist(),
                            "width": int(batch["raw_scene"][i].shape[2]),
                            "height": int(batch["raw_scene"][i].shape[1]),
                            "instance": []
                        }
                        predictions[str(img_id)]["instance"].append(
                            {
                                "image_id": img_id,
                                "category_id": category_id,
                                "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
                                "score": 1.0,
                                "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
                                "bbox3D": [
                                    output_dict["pred_pts_3d"][i][0].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][4].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][6].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][2].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][1].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][5].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][7].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][3].cpu().numpy().tolist()
                                ],
                                "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
                                "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
                                "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
                                "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
                            }
                        )
                    else:
                        predictions[str(img_id)]["instance"].append(
                            {
                                "image_id": img_id,
                                "category_id": category_id,
                                "bbox": batch["gt_bbox_2d"][i].cpu().numpy().tolist(),
                                "score": 1.0,
                                "depth": output_dict["pred_trans"][i].cpu().numpy().tolist()[2],
                                "bbox3D": [
                                    output_dict["pred_pts_3d"][i][0].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][4].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][6].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][2].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][1].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][5].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][7].cpu().numpy().tolist(),
                                    output_dict["pred_pts_3d"][i][3].cpu().numpy().tolist()
                                ],
                                "center_cam": output_dict["pred_trans"][i].cpu().numpy().tolist(),
                                "center_2D": batch["bbox_center"][i].cpu().numpy().tolist(),
                                "dimensions": output_dict["pred_dims"][i].cpu().numpy().tolist(),
                                "pose": output_dict["pred_ego_rot"][i].cpu().numpy().tolist()
                            }
                        )
                else:
                    continue

    for key, value in predictions.items():
        predictions_list.append(value)

    torch.save(predictions_list, "logs/inference_pth/instances_predictions_attn_rt.pth")


if __name__ == '__main__':
    main()