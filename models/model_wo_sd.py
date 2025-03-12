import os
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from .modules import MLP_3D_POS, FeatureDownsampler, ResidualAttentionBlock, DropBlock2D, LinearScheduler
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


class OV9D(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        embed_dim = args.embed_dim
        text_dim = 768
        dino_dim = __DINO_DIM__[args.dino_type]
        
        channels_in = dino_dim
        channels_out = embed_dim # 128

        self.embed_dim = embed_dim
        self.dino_dim = dino_dim
        
        class_embeddings = torch.load(os.path.join(args.data_path, f'oo3d9dsingle_class_embeddings.pth'))
        self.register_buffer('class_embeddings', class_embeddings)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.cls_fc = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim)
            )
        
        # self.feat_2d_downsample = FeatureDownsampler(device='cuda', dtype=torch.float32, emb_feat=3, width=pos_dim+dino_dim)
        # self.feat_3d_mlp = MLP_3D_POS(device='cuda', dtype=torch.float32, in_feature=3, width=pos_dim)
        self.attention = nn.ModuleList()
        for _ in range(args.attn_depth):
            self.attention.append(ResidualAttentionBlock(device='cuda', dtype=torch.float32, width=dino_dim, heads=8, init_scale=0.25))

        self.norm = nn.LayerNorm(dino_dim)

        if args.dino:
            self.decoder = Decoder(channels_in, channels_out, args)
            self.dino = torch.hub.load('facebookresearch/dinov2', __DINO_MODEL__[args.dino_type])
            for p in self.dino.parameters():
                p.requires_grad = False
            self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), False)
        else:
            self.decoder = Decoder(channels_in, channels_out, args)
            self.dino = None
        self.decoder.init_weights()
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.GELU())
        for m in self.last_conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

        nocs_type = args.nocs_type
        assert nocs_type in ['CE', 'L1'], f"The current nocs type {nocs_type} is not one of (CE, L1)!!"
        if nocs_type == "CE":
            self.mlp = MLP(device='cuda', dtype=torch.float32, in_channel=channels_out, out_channel=(args.nocs_bin)*3)
            pnp_cin = 2 + (args.nocs_bin-1)*3
        if nocs_type == "L1":
            self.mlp = MLP(device='cuda', dtype=torch.float32, in_channel=channels_out, out_channel=3)
            pnp_cin = 5

        self.mlp_offset = nn.Sequential(
            nn.Conv2d(embed_dim, 3, 1),
            nn.GELU(),
            nn.Conv2d(3, 3, 1)
        )
        
        self.pnp_net = ConvPnPNet(nIn=pnp_cin, rot_dim=args.rot_dim)


    def forward(self, x, feat_2d_bp, mask, roi_coord_2d, class_ids=None):    
        # import pdb; pdb.set_trace() 
        b, c, h, w = x.shape # (b, 3, 480, 480)
        # n, d = feat_3d.shape # (1024, 387)

        # feat_2d_bp = feat_2d_bp.permute(0, 3, 1, 2) # (b, 3, 480, 480)
        # feat_2d = self.feat_2d_downsample(feat_2d_bp)

        if self.dino:
            with torch.no_grad():
                pad_h, pad_w = torch.ceil(torch.tensor(h/14)).long()*14, torch.ceil(torch.tensor(w/14)).long()*14 # (490, 490)
                pad_x = torch.zeros((b, c, pad_h, pad_w)).to(x)
                pad_x[:, :, 0:h, 0:w] = x
                pad_x = (pad_x - self.pixel_mean) / self.pixel_std
                dino_feature = self.dino.get_intermediate_layers(pad_x, 
                                                                 n=self.dino.n_blocks,
                                                                 reshape=True)[-1] # (b, 1024, 35, 35), 35 = 490 // 14
                range_h = (torch.arange(h//32) * 32) / pad_h * 2 - 1
                range_w = (torch.arange(w//32) * 32) / pad_w * 2 - 1
                grid_h, grid_w = torch.meshgrid(range_h, range_w, indexing='ij')
                grid = torch.stack([grid_w, grid_h], dim=-1)
                grid = torch.stack([grid]*b, dim=0).to(x.device)
                dino_feature = torch.nn.functional.grid_sample(dino_feature, grid, align_corners=True) # (b, 1024, 15, 15)
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        # class_embeddings = self.class_embeddings[class_ids.tolist()] if class_ids is not None else self.class_embeddings
        # conv_feats = class_embeddings + self.cls_fc(class_embeddings) * self.gamma # (B, 768)
        # conv_feats = conv_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dino_feature.shape[-2], dino_feature.shape[-1]) # (B, 768, 15, 15)   

        # if self.dino:
        #     conv_feats = torch.cat([conv_feats, dino_feature], dim=1) # (B, 1024+768, 15, 15)

        dino_h, dino_w = dino_feature.shape[-2:]
        mask_resized = transforms.Resize((dino_h, dino_w))(mask).reshape(b, -1) # (b, 225)
        hidden_feature = dino_feature.reshape(b, self.dino_dim, -1).permute(0, 2, 1) # (b, 225, 1024)
        for block in self.attention:
            hidden_feature = block(hidden_feature, key_mask=mask_resized)
        hidden_feature = self.norm(hidden_feature)
        # hidden_feature = self.mlp(hidden_feature)
        # hidden_feature = self.norm2(hidden_feature)

        conv_feats = hidden_feature.permute(0, 2, 1).reshape(b, self.dino_dim, dino_h, dino_w)
        
        out = self.decoder([conv_feats])
        out_feat = self.last_conv(out) # (B, 128, 480, 480)

        out_feat_offset = self.mlp_offset(out_feat) # (B, 3, 480, 480)
        out_feat_logits = self.mlp(out_feat) # (B, 128, 480, 480)

        assert out_feat_logits.shape[1] % 3 == 0
        out_x, out_y, out_z = torch.split(out_feat_logits, out_feat.shape[1] // 3, dim=1)
        if out_x.shape[1] > 1 and out_y.shape[1] > 1 and out_z.shape[1] > 1:
            out_x_softmax = F.softmax(out_x[:, :-1, :, :], dim=1)
            out_y_softmax = F.softmax(out_y[:, :-1, :, :], dim=1)
            out_z_softmax = F.softmax(out_z[:, :-1, :, :], dim=1)
            out_nocs_feat = torch.cat([out_x_softmax, out_y_softmax, out_z_softmax], dim=1) # (b, (bin-1)*3, 480,480)
        else:
            out_nocs_feat = torch.cat([out_x, out_y, out_z], dim=1) # (b, 3, 480, 480)
        
        out_nocs = torch.cat([out_x, out_y, out_z], dim=1) # (b, bin*3, 480,480) or (b, 3, 480, 480)

        pnp_inp = torch.cat([out_nocs_feat, roi_coord_2d], dim=1) # (B, c, 480, 480)
        pred_rot_, pred_t_ = self.pnp_net(pnp_inp) # pred_rot_ (B, 4), pred_t_ (B, 3)
        if pred_rot_.shape[-1] == 4:
            pred_rot_m = quat2mat_torch(pred_rot_) # (B, 3, 3)
        elif pred_rot_.shape[-1] == 6:
            pred_rot_m = ortho6d_to_mat_batch(pred_rot_) # (B, 3, 3)
        else:
            raise ValueError(f"The dimension of pred_r as {pred_rot_.shape[-1]} is not in (4, 6)")

        return {'pred_nocs_feat': out_nocs, 'pred_nocs_offset': torch.tanh(out_feat_offset)*(0.5/self.args.nocs_bin), 'pred_r': pred_rot_m, 'pred_t': pred_t_}
    

class MLP(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, in_channel: int, out_channel: int):
        super().__init__()
        self.c_fc = nn.Conv2d(in_channel, out_channel*2, kernel_size=1, device=device, dtype=dtype)
        self.c_mid = nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, device=device, dtype=dtype)
        self.c_proj = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, device=device, dtype=dtype)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_mid(self.gelu(self.c_fc(x)))))


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
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # import pdb; pdb.set_trace()
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

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
        num_gn_groups=32,
        drop_prob=0.0,
        dropblock_size=5,
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
            self.features.append(nn.Conv2d(_in_channels, featdim, kernel_size=6, stride=4, padding=padding, bias=False))
            self.features.append(nn.GroupNorm(num_gn_groups, featdim))
            self.features.append(nn.ReLU(inplace=True))
        for i in range(num_layers - 3):  # when num_layers > 3
            self.features.append(nn.Conv2d(featdim, featdim, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.GroupNorm(num_gn_groups, featdim))
            self.features.append(nn.ReLU(inplace=True))

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(featdim * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(256, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # feature for extent
        # self.extent_fc1 = nn.Linear(3, 64)
        # self.extent_fc2 = nn.Linear(64, 128)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

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
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)
        return rot, t