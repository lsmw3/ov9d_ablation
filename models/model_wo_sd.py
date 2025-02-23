import os
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import torch.nn.functional as F


class OV9D(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        embed_dim = 96
        text_dim = 768
        
        channels_in = embed_dim*8 # 768
        channels_out = embed_dim # 96
        
        class_embeddings = torch.load(os.path.join(args.data_path, f'{args.data_name}_class_embeddings.pth'))
        self.register_buffer('class_embeddings', class_embeddings)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        self.cls_fc = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim)
            )

        if args.dino:
            self.decoder = Decoder(channels_in+1024, channels_out, args)
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            for p in self.dino.parameters():
                p.requires_grad = False
            self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), False)
        else:
            self.decoder = Decoder(channels_in, channels_out, args)
            self.dino = None
        self.decoder.init_weights()
        
        self.last_layer_nocs = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 3, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_nocs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x, class_ids=None):    
        # import pdb; pdb.set_trace() 
        b, c, h, w = x.shape # (b, 3, 480, 480)
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

        class_embeddings = self.class_embeddings[class_ids.tolist()] if class_ids is not None else self.class_embeddings
        conv_feats = class_embeddings + self.cls_fc(class_embeddings) * self.gamma # (B, 768)
        conv_feats = conv_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, dino_feature.shape[-2], dino_feature.shape[-1]) # (B, 768, 15, 15)   

        if self.dino:
            conv_feats = torch.cat([conv_feats, dino_feature], dim=1) # (B, 1024+768, 15, 15)
        
        out = self.decoder([conv_feats])
        out_nocs = self.last_layer_nocs(out) # (B, 3, 480, 480)

        return {'pred_nocs': out_nocs}


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

