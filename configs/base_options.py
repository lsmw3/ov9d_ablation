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


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='/data/ssd1/')
        parser.add_argument('--dataset',      type=str, default='nyudepthv2')
        parser.add_argument('--batch_size',   type=int, default=8)
        parser.add_argument('--workers',      type=int, default=8)
        
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)
        parser.add_argument('--num_filters', nargs='+', type=int)
        parser.add_argument('--deconv_kernels', nargs='+', type=int)

        parser.add_argument('--dino', action='store_true') 
        parser.add_argument('--dino_type', type=str, default=None)
        parser.add_argument('--attn_depth', type=int, default=4)

        parser.add_argument('--nocs_bin', type=int, default=32)

        parser.add_argument('--rot_dim', type=int, default=6)
        parser.add_argument('--embed_dim', type=int, default=128)
        parser.add_argument('--latent_size', type=int, default=32)
        parser.add_argument('--scale_size',  type=int, default=490)
        parser.add_argument('--virtual_focal',  type=float, default=512.0)
        
        parser.add_argument('--num_gpus', type=int, default=1)

        parser.add_argument('--decode_rt', action='store_true')
        parser.add_argument('--with_attn', action='store_true')

        parser.add_argument('--ckpt_path', type=str, default=None)
        
        return parser
