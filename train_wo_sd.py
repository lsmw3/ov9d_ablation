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

import wandb


def main():
    wandb.init(project="ov9d")

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
        'data_type': args.data_train, 
    }
    dataset_kwargs['scale_size'] = args.scale_size

    train_dataset = get_dataset(**dataset_kwargs)
    dataset_kwargs['data_type'] = args.data_val
    dataset_kwargs['num_view'] = 50
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_train = RandomSampler(train_dataset)

    sampler_val = SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=sampler_train,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            sampler=sampler_val,
                            pin_memory=True)
    
    project_name = "ov9d"
    exp_name = "ov9d_ablation"

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    logger = WandbLogger(name=exp_name,project=project_name, save_dir="./wandb", entity="large-reconstruction-model")

    # device = torch.device(args.gpu)
    nproc_per_node = int(os.getenv('NPROC_PER_NODE', '1'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    strategy = DDPStrategy(find_unused_parameters=True) if world_size > 1 else None

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "ov9d_ablation"),
        filename='epoch_{epoch}',
        monitor='val/loss',
        # save_last=True,
        save_top_k=3,             # Set to -1 to save all checkpoints
        every_n_epochs=50,
        save_on_train_epoch_end=True
    )

    my_trainer = CustomTrainer(args)

    trainer = L.Trainer(devices=nproc_per_node,
                        num_nodes=1,
                        max_epochs=args.epochs,
                        # max_epochs=1,
                        accelerator='gpu',
                        strategy=strategy,
                        accumulate_grad_batches=1,
                        logger=logger,
                        gradient_clip_val=0.5,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=1,
                        limit_val_batches=1.,  # Run on only 10% of the validation data
                        limit_train_batches=1.,
                        )
    
    t0 = datetime.now()
    
    trainer.fit(
        my_trainer, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        )
    
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))


if __name__ == '__main__':
    main()
