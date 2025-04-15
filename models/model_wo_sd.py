import os
import math
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from .modules import MLP_3D_POS, FeatureDownsampler, ResidualAttentionBlock, DropBlock2D, LinearScheduler
from .retrieval_attention import ViewTransformer
from utils.pose_utils import quat2mat_torch, ortho6d_to_mat_batch


__DINO_DIM__ = {
    'small': 384,
    'base': 768,
    'large': 1024
}
__DINO_MODEL__ = {
    'small': 'dinov2_vits14',
    'base': 'dinov2_vitb14',
    'large': 'dinov2_vitl14'
}


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class OV9D(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        embed_dim = args.embed_dim # 128
        text_dim = 768
        dino_dim = __DINO_DIM__[args.dino_type]

        self.embed_dim = embed_dim
        self.dino_dim = dino_dim
        
        # class_embeddings = torch.load(os.path.join(args.data_path, f'oo3d9dsingle_class_embeddings.pth'))
        # self.register_buffer('class_embeddings', class_embeddings)
        # self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        # self.cls_fc = nn.Sequential(
        #         nn.Linear(text_dim, text_dim),
        #         nn.GELU(),
        #         nn.Linear(text_dim, text_dim)
        #     )
        
        # self.feat_2d_downsample = FeatureDownsampler(device='cuda', dtype=torch.float32, emb_feat=3, width=pos_dim+dino_dim)
        # self.feat_3d_mlp = MLP_3D_POS(device='cuda', dtype=torch.float32, in_feature=3, width=pos_dim)
        self.attention = nn.ModuleList()
        for _ in range(args.attn_depth):
            # self.attention.append(ResidualAttentionBlock(device='cuda', dtype=torch.float32, width=dino_dim, heads=8, init_scale=0.25))
            self.attention.append(ViewTransformer(width=embed_dim, attn_depth=args.attn_depth))

        self.norm = nn.LayerNorm(embed_dim)

        if args.dino:
            self.decoder = Decoder_MLP(device='cuda', dtype=torch.float32, in_channel=embed_dim, out_channel=3)
            self.dino = torch.hub.load('facebookresearch/dinov2', __DINO_MODEL__[args.dino_type])
            for p in self.dino.parameters():
                p.requires_grad = False
            self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1), False)
            self.dino_embed = nn.Linear(dino_dim, embed_dim)
        else:
            self.decoder = Decoder_MLP(device='cuda', dtype=torch.float32, in_channel=dino_dim, out_channel=3)
            self.dino = None
        
        self.pnp_net = ConvPnPNet(nIn=5, featdim=embed_dim, rot_dim=args.rot_dim, num_layers=6)

        if args.decode_rt:
            self.rt_decoder = Decoder_RT(dtype=torch.float32, feat_dim=embed_dim, rot_dim=args.rot_dim, conv_dims=[128, 128, 128], mid_dims=[1024, 256, 128])
            
        # self.apply(init_weights)


    def forward(self, x, mask, roi_coord_2d):
        # import pdb; pdb.set_trace() 
        b, c, h, w = x.shape # (b, 3, h, w)
        # n, d = feat_3d.shape # (1024, 387)

        # feat_2d_bp = feat_2d_bp.permute(0, 3, 1, 2) # (b, 3, 480, 480)
        # feat_2d = self.feat_2d_downsample(feat_2d_bp)

        if self.dino is not None:
            with torch.no_grad():
                x = (x - self.pixel_mean) / self.pixel_std
                dino_feature = self.dino.get_intermediate_layers(x, 
                                                                 n=self.dino.n_blocks,
                                                                 reshape=True)[-1] # (b, 384, 35, 35), 35 = 490 // 14

        # class_embeddings = self.class_embeddings[class_ids.tolist()] if class_ids is not None else self.class_embeddings
        # conv_feats = class_embeddings + self.cls_fc(class_embeddings) * self.gamma # (B, 768)
        # conv_feats = conv_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dino_feature.shape[-2], dino_feature.shape[-1]) # (B, 768, 15, 15)   

        # if self.dino:
        #     conv_feats = torch.cat([conv_feats, dino_feature], dim=1) # (B, 1024+768, 15, 15)

        dino_h, dino_w = dino_feature.shape[-2:] # (35, 35)
        hidden_feature = dino_feature.reshape(b, self.dino_dim, -1).permute(0, 2, 1) # (b, 35*35, 384)
        hidden_feature = self.dino_embed(hidden_feature) # (b, 35*35, 128)

        if self.args.with_attn:
            for block in self.attention:
                hidden_feature = block(hidden_feature)

        conv_feats = hidden_feature.permute(0, 2, 1).reshape(b, self.embed_dim, dino_h, dino_w)

        if self.args.decode_rt:
            pred_rot_, pred_t_, pred_dims = self.rt_decoder(conv_feats)
            if pred_rot_.shape[-1] == 4:
                pred_rot_m = quat2mat_torch(pred_rot_) # (B, 3, 3)
            elif pred_rot_.shape[-1] == 6:
                pred_rot_m = ortho6d_to_mat_batch(pred_rot_) # (B, 3, 3)
            else:
                raise ValueError(f"The dimension of pred_r as {pred_rot_.shape[-1]} is not in (4, 6)")
            out_dict = {'pred_r': pred_rot_m, 'pred_t': pred_t_, 'pred_dims': pred_dims}
            return out_dict
        
        out = self.decoder(hidden_feature).permute(0, 2, 1).reshape(b, 3, dino_h, dino_w)
        out_feat_up = F.interpolate(out, (h, w), mode='bicubic', align_corners=True)
        
        out_nocs = out
        out_nocs_feat = out_feat_up

        pnp_inp = torch.cat([out_nocs_feat, roi_coord_2d], dim=1) * mask.unsqueeze(1) # (B, c, h, w)
        pred_rot_, pred_t_, pred_dims = self.pnp_net(pnp_inp) # pred_rot_ (B, 4), pred_t_ (B, 3)
        if pred_rot_.shape[-1] == 4:
            pred_rot_m = quat2mat_torch(pred_rot_) # (B, 3, 3)
        elif pred_rot_.shape[-1] == 6:
            pred_rot_m = ortho6d_to_mat_batch(pred_rot_) # (B, 3, 3)
        else:
            raise ValueError(f"The dimension of pred_r as {pred_rot_.shape[-1]} is not in (4, 6)")
        
        out_dict = {'pred_nocs_feat': out_nocs, 'pred_nocs_ori_size': out_feat_up, 'pred_r': pred_rot_m, 'pred_t': pred_t_, 'pred_dims': pred_dims}

        return out_dict
    

class Decoder_MLP(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, in_channel: int, out_channel: int):
        super().__init__()
        self.c_fc = nn.Linear(in_channel, out_channel*2, device=device, dtype=dtype)
        self.c_mid = nn.Linear(out_channel*2, out_channel*2, device=device, dtype=dtype)
        self.c_proj = nn.Linear(out_channel*2, out_channel, device=device, dtype=dtype)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.c_proj(self.act(self.c_mid(self.act(self.c_fc(x)))))


class Decoder_RT(nn.Module):
    def __init__(self, dtype: torch.dtype, feat_dim: int, rot_dim: int, conv_dims: list, mid_dims: list):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        last_dim = feat_dim
        self.middle_dim = conv_dims[-1]

        for i in range(len(conv_dims)):
            out_dim = conv_dims[i]
            if i == 0:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(last_dim, out_dim, kernel_size=7, stride=5, padding=1),
                        # nn.BatchNorm2d(out_dim),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
            else:
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(last_dim, out_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(out_dim),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
            last_dim = out_dim

        last_dim = last_dim * 7 * 7

        for i in range(len(mid_dims)):
            out_dim = mid_dims[i]
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(last_dim, out_dim, dtype=dtype),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            )
            last_dim = out_dim

        self.fc_r = nn.Linear(last_dim, rot_dim, dtype=dtype)
        self.fc_uv = nn.Linear(last_dim, 2, dtype=dtype)
        self.fc_z = nn.Linear(last_dim, 1, dtype=dtype)
        self.fc_dims = nn.Linear(last_dim, 3, dtype=dtype)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(-1, self.middle_dim * 7 * 7)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)

        r = self.fc_r(x)
        uv = self.fc_uv(x)
        z = self.fc_z(x)
        t = torch.cat([uv, z], dim=-1)
        dims = self.fc_dims(x)

        return r, t, dims


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels

        # import pdb; pdb.set_trace()
        
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv, # 3
            args.num_filters, # 32 32 32
            args.deconv_kernels, # 2 2 2
        )

        self.mid_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=args.num_filters[-1],
                out_channels=args.num_filters[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(args.num_filters[-1]),
            nn.ReLU(inplace=True)
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
    

    def forward(self, conv_feats, h, w):
        # import pdb; pdb.set_trace()
        out = self.deconv_layers(conv_feats)
        out = self.mid_conv_layers(out)
        out = F.interpolate(out, size=(h, w), mode="bicubic", align_corners=True)
        out = self.conv_layers(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


class ConvPnPNet(nn.Module):
    def __init__(
        self,
        nIn,
        featdim=128,
        rot_dim=6,
        num_layers=3,
        num_gn_groups=8,
        drop_prob=0.0,
        dropblock_size=5,
        dtype=torch.float32
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
        """
        super().__init__()
        self.featdim = featdim

        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        assert num_layers >= 3, num_layers
        self.features = nn.ModuleList()
        for i in range(3):
            _in_channels = nIn if i == 0 else featdim
            padding = 2 if i == 2 else 1
            self.features.append(nn.Conv2d(_in_channels, featdim, kernel_size=6, stride=4, padding=padding, bias=False, dtype=dtype))
            self.features.append(nn.GroupNorm(num_gn_groups, featdim, dtype=dtype))
            self.features.append(nn.ReLU(inplace=True))
        for i in range(num_layers - 3):  # when num_layers > 3
            self.features.append(nn.Conv2d(featdim, featdim, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype))
            self.features.append(nn.GroupNorm(num_gn_groups, featdim, dtype=dtype))
            self.features.append(nn.ReLU(inplace=True))

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(featdim * 8 * 8, 1024, dtype=dtype)
        self.fc2 = nn.Linear(1024, 256, dtype=dtype)
        # TODO: predict centroid and z separately
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.fc_r = nn.Linear(256, rot_dim, dtype=dtype)
        self.fc_uv = nn.Linear(256, 2, dtype=dtype)
        self.fc_z = nn.Linear(256, 1, dtype=dtype)
        self.fc_dims = nn.Linear(256, 3, dtype=dtype)

    def forward(self, coor_feat):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape

        x = coor_feat

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        x = x.view(-1, self.featdim * 8 * 8)
        # extent feature
        # # TODO: use extent the other way: denormalize coords
        # x_extent = self.act(self.extent_fc1(extents))
        # x_extent = self.act(self.extent_fc2(x_extent))
        # x = torch.cat([x, x_extent], dim=1)
        #
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        r = self.fc_r(x)
        uv = self.fc_uv(x)
        z = self.fc_z(x)
        t = torch.cat([uv, z], dim=-1)
        dims = self.fc_dims(x)

        return r, t, dims