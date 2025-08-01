from configs.base_options import BaseOptions
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        parser.add_argument('--data_name', type=str, default='ycbv')
        parser.add_argument('--data_train', type=str, default='train')
        parser.add_argument('--data_val', type=str, default='val')
        parser.add_argument('--data_3d_feat', type=str, default=None)
        parser.add_argument('--ckpt', type=str, default=None)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=25)
        parser.add_argument('--lr',          type=float, default=1e-4)
        parser.add_argument('--min_lr',          type=float, default=1e-4)
        parser.add_argument('--weight_decay',          type=float, default=5e-2)
        parser.add_argument('--layer_decay',          type=float, default=0.9)
        
        parser.add_argument('--log_dir', type=str, default='./logs')

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--pro_bar', type=str2bool, default='False')
        parser.add_argument('--save_freq', type=int, default=1)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--save_model', action='store_true')     
        parser.add_argument(
            '--resume-from', help='the checkpoint file to resume from')
        parser.add_argument('--auto_resume', action='store_true')   
        parser.add_argument('--save_result', action='store_true')

        return parser
