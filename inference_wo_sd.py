import os
import numpy as np
from datetime import datetime

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

    inference_dataset = get_dataset(**dataset_kwargs, is_train=False)
    # sampler_val = SequentialSampler(inference_dataset)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    
    # project_name = "ov9d"
    # exp_name = "ov9d_ablation"

    # os.environ["WANDB__SERVICE_WAIT"] = "300"
    # logger = WandbLogger(name=exp_name,project=project_name, save_dir="./wandb", entity="large-reconstruction-model")

    # device = torch.device(args.gpu)
    # nproc_per_node = int(os.getenv('NPROC_PER_NODE', '1'))
    # world_size = int(os.getenv('WORLD_SIZE', '1'))
    # strategy = DDPStrategy(find_unused_parameters=True) if world_size > 1 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomTrainer(args=args)
    model = model.load_from_checkpoint("logs/debugs/nan_detected_303.ckpt", args=args)
    model = model.to(device)

    num_instances = len(inference_loader)
    num_nan = 0
    with torch.no_grad():
        for idx, batch in enumerate(inference_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            output = model(batch)
            if torch.isnan(output):
                num_nan += 1
        print(f"{num_nan} NAN value from {num_instances} objects")
            # predicted_class = torch.argmax(output, dim=1)
            # print(f"Sample {idx}: Predicted class {predicted_class.item()}")


if __name__ == '__main__':
    main()