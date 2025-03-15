import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featup.layers import ChannelNorm


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP_3D_POS(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, in_feature: int, width: int, init_scale: float = 0.25):
        super().__init__()
        self.width = width
        self.c_in = nn.Linear(in_feature, width, device=device, dtype=dtype)
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_in, init_scale)
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(self.gelu(self.c_in(x)))))
    

class FeatureDownsampler(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, emb_feat: int = 3, width: int = 128+384):
        super().__init__()

        self.down_1 = nn.Conv2d(in_channels=emb_feat, out_channels=width, kernel_size=6, stride=4, padding=1, device=device, dtype=dtype)
        self.batch_norm_1 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.down_2 = nn.Conv2d(in_channels=width, out_channels=width*4, kernel_size=6, stride=4, padding=1, groups=width, device=device, dtype=dtype)
        self.batch_norm_2 = nn.BatchNorm2d(width*4, device=device, dtype=dtype)

        self.down_3 = nn.Conv2d(in_channels=width*4, out_channels=width, kernel_size=4, stride=2, padding=1, groups=width, device=device, dtype=dtype)
        self.batch_norm_3 = nn.BatchNorm2d(width, device=device, dtype=dtype)

        self.gelu = nn.GELU()

    def forward(self, x):
        h_1 = self.gelu(self.batch_norm_1(self.down_1(x)))
        h_2 = self.gelu(self.batch_norm_2(self.down_2(h_1)))
        h_3 = self.gelu(self.batch_norm_3(self.down_3(h_2)))

        return h_3
    

class MLP(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype, heads: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads

    def forward(self, qkv, query_mask=None, key_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
        if key_mask is not None:
            weight = weight.masked_fill(~key_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        if query_mask is not None:
            weight = weight.masked_fill(~query_mask.unsqueeze(1).unsqueeze(-1), float('-inf'))
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)
    

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width*3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, query_mask=None, key_mask=None):
        qkv = self.c_qkv(x)
        x = self.attention(qkv, query_mask=query_mask, key_mask=key_mask)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, query_mask: torch.Tensor = None, key_mask: torch.Tensor = None):
        x = x + self.attn(self.ln_1(x), query_mask=query_mask, key_mask=key_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Featup(nn.Module):
    def __init__(self, use_norm=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load original model
        self.model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=False).to(device)
        # Create separate normalization layer
        self.channel_norm = ChannelNorm(384) if use_norm else nn.Identity()
        
    def forward(self, x):
        return self.model.upsampler(self.get_patch_token(x), x)
    
    def get_patch_token(self, x):
        features = self.model.model(x)  # Get features including CLS token
        # Apply normalization
        features = self.channel_norm(features)
        return features
    
    def get_feat(self, x):
        batch_size = x.shape[0]
        patch_token = self.model.model(x).permute(0,2,3,1).reshape(batch_size,-1,384)
        cls_token = self.model.model.get_cls_token(x).unsqueeze(1)
        features = torch.cat([cls_token, patch_token], dim=1)
        norm = torch.linalg.norm(features, dim=-1)[:, :, None]
        features = features / norm
        patch_token = features[:,1:,:].permute(0,2,1).reshape(batch_size,384,16,16)
        cls_token = features[:,0,:]

        return patch_token, cls_token
    

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
    

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1