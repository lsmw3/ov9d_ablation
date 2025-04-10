import os
import numpy as np
import re
import pickle
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


class SaveOnNanCallback(ModelCheckpoint):
    def __init__(self, dirpath="logs/debugs", filename="nan_detected_{step}"):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if outputs contain NaN values
        if outputs is not None:
            # Assuming outputs is a dictionary with 'loss' key
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                # If outputs is just the loss tensor
                loss = outputs
                
            if trainer.global_step >= 300:
                if torch.isnan(loss).any():
                    # NaN detected, save checkpoint
                    checkpoint_path = f"{self.dirpath}/{self.filename.format(step=trainer.global_step)}.ckpt"
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"NaN detected at step {trainer.global_step}. Checkpoint saved to {checkpoint_path}")
                else:
                    checkpoint_path = f"logs/steps_before_nan/{self.filename.format(step=trainer.global_step)}.ckpt"
                    trainer.save_checkpoint(checkpoint_path)


class WeightCheckCallback(L.Callback):
    def __init__(self, args, grad_threshold_min=1e-6, grad_threshold_max=1e4):
        super().__init__()
        self.args = args
        self.grad_threshold_min = grad_threshold_min
        self.grad_threshold_max = grad_threshold_max

        self.gradient_records = []
    
    # def on_train_epoch_end(self, trainer, pl_module):
    #     for name, param in pl_module.named_parameters():
    #         if torch.isnan(param).any():
    #             print(f"NaN detected in {name} after epoch {trainer.current_epoch}")
    #             # You can also raise an exception or take other actions here

    def clean_numbered_files(self, folder_path, pattern):
        """
        Deletes all but the two most recent (largest number) files in a folder.

        :param folder_path: Path to the folder containing files.
        :param pattern: Regex pattern to match filenames and extract numbers.
        """
        regex = re.compile(pattern)
        
        # Find all matching files and extract their numbers
        files = []
        for filename in os.listdir(folder_path):
            match = regex.match(filename)
            if match:
                file_number = int(match.group(1))  # Extract the number
                files.append((file_number, os.path.join(folder_path, filename)))

        # Sort files by extracted number in descending order
        files.sort(reverse=True, key=lambda x: x[0])

        # If more than 2 files exist, delete all except the last two
        if len(files) > 10:
            for _, file_path in files[2:]:  # Keep the first two, delete the rest
                os.remove(file_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for key, value in pl_module.loss_dict.items():
            if torch.isnan(torch.tensor([value])):
                print(f"Detect nan loss value of {key} at step {trainer.global_step}")
                # trainer.should_stop = True
                # break

        gradients = {f"{trainer.global_step}": {name: param.grad.cpu().numpy() for name, param in pl_module.named_parameters() if param.grad is not None}}
        gradients.update({f"{trainer.global_step}_inputs": {key: value.detach().cpu().numpy() for key, value in batch.items() if isinstance(value, torch.Tensor)}})
        gradients.update({f"{trainer.global_step}_outputs": {key: value.detach().cpu().numpy() for key, value in pl_module.model_output.items() if value is not None}})
        self.gradient_records.append(gradients)
        if len(self.gradient_records) >= 10:
            self.gradient_records = self.gradient_records[-2:]

        ckpt_folder = "logs/debug_ckpt_w_sz" if self.args.predict_dims else "logs/debug_ckpt"
        checkpoint_path = os.path.join(ckpt_folder, f"{trainer.global_step}.ckpt")
        trainer.save_checkpoint(checkpoint_path)
        self.clean_numbered_files(ckpt_folder, r"(\d+)\.ckpt")

        grad_folder = "logs/debug_grad_w_sz" if self.args.predict_dims else "logs/debug_grad"
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.max(grad) > self.grad_threshold_max:
                    print(f"Gradient supervision alert: {name} has extreme big values {torch.max(grad)} after step {trainer.global_step}")
                # if torch.min(torch.abs(grad)) < self.grad_threshold_min:
                #     print(f"Gradient supervision alert: {name} has extreme small values {torch.min(torch.abs(grad))} after batch {batch_idx}")
                    # trainer.should_stop = True
                # np.savez(f"logs/debug_grad/grad_{trainer.global_step}.npz", **gradients)
                # clean_numbered_files("logs/debug_grad", r"grad_(\d+)\.npz")
            if torch.isnan(param).any():
                print(f"NaN detected in {name} after step {trainer.global_step}")
                trainer.should_stop = True
                # gradients = {name: param.grad.cpu().numpy() for name, param in pl_module.named_parameters() if param.grad is not None}
                # print(gradients)
                with open(os.path.join(grad_folder, 'grad_data_records.pkl'), 'wb') as f:
                    pickle.dump(self.gradient_records, f)
                break


class NaNGradientSkipper(L.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx):
        # Check for NaN gradients in all parameters
        skip_optimization = False
        for key, value in pl_module.loss_dict.items():
            if torch.isnan(torch.tensor([value])):
                print(f"Detect nan loss value of {key} at step {trainer.global_step}, skipping optimization step")
                skip_optimization = True
                break
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name} at step {trainer.global_step}, skipping optimization step")
                skip_optimization = True
                break
                
        if skip_optimization:
            # Zero out all gradients to prevent the optimization step
            optimizer.zero_grad()
            # Signal to skip the optimization step
            return -1


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
        'feat_3d_path': args.data_3d_feat ,
        'xyz_bin': args.nocs_bin,
        'raw_w': args.raw_w,
        'raw_h': args.raw_h
    }
    dataset_kwargs['scale_size'] = args.scale_size

    train_dataset = get_dataset(**dataset_kwargs)
    dataset_kwargs['data_type'] = args.data_val
    dataset_kwargs['num_view'] = 50
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_train = RandomSampler(train_dataset)

    sampler_val = RandomSampler(val_dataset) # SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=sampler_train,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=sampler_val,
                            num_workers=args.workers,
                            pin_memory=True)
    
    project_name = "ov9d"
    exp_name = "ov9d_ablation"

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    logger = WandbLogger(name=exp_name,project=project_name, save_dir="./wandb", entity="large-reconstruction-model")

    # device = torch.device(args.gpu)
    # nproc_per_node = int(os.getenv('NPROC_PER_NODE', '1'))
    # world_size = int(os.getenv('WORLD_SIZE', '1'))
    # strategy = DDPStrategy(find_unused_parameters=True) if world_size > 1 else None

    if args.with_attn and args.decode_rt:
        ckpt_folder_name = "attn_rt"
    if args.with_attn and not args.decode_rt:
        ckpt_folder_name = "nocs_rt"
    if not args.with_attn and args.decode_rt:
        ckpt_folder_name = "rt"
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, ckpt_folder_name),
        filename='{epoch}',
        # monitor='val/loss',
        # save_last=True,
        save_top_k=-1,             # Set to -1 to save all checkpoints
        every_n_epochs=5,
        save_on_train_epoch_end=True
    )

    nan_callback = SaveOnNanCallback(dirpath=os.path.join(args.log_dir, "debugs"))

    my_trainer = CustomTrainer(args)

    trainer = L.Trainer(devices=args.num_gpus, #[0,1]
                        num_nodes=1,
                        max_epochs=args.epochs,
                        # max_epochs=1,
                        accelerator='gpu',
                        strategy=DDPStrategy(find_unused_parameters=True),
                        accumulate_grad_batches=1,
                        logger=logger,
                        gradient_clip_val=0.5,
                        # gradient_clip_algorithm="value",
                        callbacks=[checkpoint_callback, NaNGradientSkipper()], # WeightCheckCallback(args)
                        check_val_every_n_epoch=3,
                        limit_val_batches=1.,  # Run on only 10% of the validation data
                        limit_train_batches=1.,
                        )
    
    t0 = datetime.now()
    
    trainer.fit(
        my_trainer, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="logs/ov9d_ablation_rt/epoch=8-v1.ckpt"
    )
    
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))


if __name__ == '__main__':
    main()
