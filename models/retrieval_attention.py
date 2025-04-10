import copy
import torch
import torch.nn as nn
from .modules import LinearAttention, FullAttention

class ViewEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        attention="linear",
        kernel_fn="elu + 1",
        redraw_interval=1,
        d_kernel=None,
        rezero=None,
        norm_method="layernorm",
    ):
        super(ViewEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention with zero initialization
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = (
            LinearAttention(
                self.dim,
                kernel_fn=kernel_fn,
                redraw_interval=redraw_interval,
                d_kernel=d_kernel,
            )
            if attention == "linear"
            else FullAttention()
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network with zero initialization
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        if norm_method == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_method == "instancenorm":
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
        else:
            raise NotImplementedError

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if rezero is not None:
            # Initialize rezero parameter to the specified value
            self.res_weight = nn.Parameter(torch.Tensor([rezero]), requires_grad=True)
        self.rezero = True if rezero is not None else False

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.dropout1(
            message
        )  # dropout before merging multi-head queried outputs
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.dropout2(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message if not self.rezero else x + self.res_weight * message


class ViewTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, width, attn_depth):
        super(ViewTransformer, self).__init__()

        # self.config = config = config["coarse"]
        # self.d_model = config["d_model"]  # Feature of query image
        # self.nhead = config["nhead"]
        # self.layer_iter_n =  config["layer_iter_n"]

        self.width = width
        
        # self.norm_method = config["norm_method"]
        # if config["redraw_interval"] is not None:
        #     assert (
        #         config["redraw_interval"] % 2 == 0
        #     ), "redraw_interval must be divisible by 2 since each attetnion layer is repeatedly called twice."

        encoder_layer = build_encoder_layer(width, nhead=8)

        # Define the layer sequence for the block
        self.layers = nn.ModuleList()
        for _ in range(attn_depth):
            self.layers.append(encoder_layer)

        # if config["final_proj"]:
        #     self.final_proj = nn.Linear(config["d_model"], config["d_model"], bias=True)

        # self._reset_parameters()

    def _reset_parameters(self):
        for para in self.parameters():
            if para.dim() > 1:
                nn.init.xavier_uniform_(para)
 
    def forward(self, x):
        """
        Args:
           x (torch.Tensor): [N, L, C]
           desc3d_db (torch.Tensor): [N, L, C] 
           desc2d_db (torch.Tensor): [N, M, C]
           desc_2d_mask (torch.Tensor): [N, M] (optional)
           view_emb (torch.Tensor): [N, C]
        """
        self.device = x.device

        for layer_idx, layer in enumerate(self.layers):
            # 2D Cross Attention on 3D
            x = layer(x, x)

        return x


def build_encoder_layer(width, nhead):
    layer = ViewEncoderLayer(width,nhead)
    return layer