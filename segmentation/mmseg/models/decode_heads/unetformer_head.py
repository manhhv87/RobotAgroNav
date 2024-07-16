# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


# https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation *
                               (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation *
                               (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation *
                               (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


# https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Global-local attention
class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,                       # Number of input channels
                 num_heads=16,                  # Number of attention heads h
                 # If True, add a learnable bias to query, key, value. Default: False
                 qkv_bias=False,
                 window_size=8,                 # The height and width of the window w
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        # Number of input channels of each attention heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size   # Wh, Ww

        # local branch employs two parallel convolutional layers with
        # kernel sizes of 3 × 3 and 1 × 1 to extract the local context
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)

        # global branch deploys the window-based multi-head selfattention to capture global context.
        # we first use a standard 1 × 1 convolution to expand the channel dimension
        # of the input 2D feature map ∈ R (B×C×H×W) to 3 times.
        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        # cross-shaped window context interaction
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,
                                   padding=(window_size//2 - 1, 0))

        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1,
                                   padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid(
                [coords_h, coords_w]))   # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

            # 2, Wh*Ww, Wh*Ww
            relative_coords = coords_flatten[:, :,
                                             None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index",
                                 relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    # ps - padding size
    def pad(self, x, ps):
        _, _, H, W = x.size()   # B, C, H, W

        if W % ps != 0:
            # pad only the last dimension (W) of the input tensor by (0, ps - W % ps)
            x = F.pad(input=x, pad=(0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            # pad last dim by (0, 0) and 2nd to last by (0, ps - H % ps)
            x = F.pad(input=x, pad=(0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        # pad last dim by (0, 1) and 2nd to last by (0, 1)
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        """
        Args:
             x: input features with shape of (B, H, W, C)
        """
        B, C, H, W = x.shape

        # local context
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape      # Hp, Wp = High/Width after padding
        qkv = self.qkv(x)           # con 1x1 -> (B x 3C x H x W)

        # depth-to-space: reshape -> transpose -> reshape
        #
        # q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d',
        #                     h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        #
        #                                    0  1  2  3   4   5    6   7
        #   reshape:       qkv = qkv.reshape(b, 3, h, d, ws1, hh, ws2, ww)
        #   transpose：    qkv = qkv.transpose(1, 0, 5, 7, 2, 4, 6, 3) = (3, b, hh, ww, h, ws1, ws2, d)
        #   reshape:   q, k, v = qkv.reshape(3, b*hh*ww, h, ws1*ws2, d)
        #
        # q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
        #                     h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        #
        #                                    0  1  2  3   4   5   6    7
        #   reshape:       qkv = qkv.reshape(b, 3, h, d, hh, ws1, ww, ws2)
        #                        --> window partition --> (B x H/w x W/w) x 3C x w x w
        #   transpose：    qkv = qkv.transpose(1, 0, 4, 6, 2, 5, 7, 3) = (3, b, hh, ww, h, ws1, ws2, d)
        #                        --> reshape each head --> 1D sequence --> (3 x B x H/w x W/w x h) x (w x w) x C/h
        #   reshape:   q, k, v = qkv.reshape(3, b*hh*ww, h, ws1*ws2, d)
        #                        --> assign to each head --> (B x H/w x W/w) x (w x w) x C/h
        #
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
                            h=self.num_heads, d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws,
                            qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # wh*ww, wh*ww, number of heads
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # number of heads, wh*ww, wh*ww

            # Calculate the position code and add it together with the attention
            dots += relative_position_bias.unsqueeze(0)

        # Do softmax on the score of the attention mechanism
        attn = dots.softmax(dim=-1)
        attn = attn @ v

        # attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
        #                  h=self.num_heads, d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws,
        #                  ws1=self.ws, ws2=self.ws)
        #
        #                                  0  1   2   3   4    5   6
        #   reshape:   attn = attn.reshape(b, hh, ww, h, ws1, ws2, d)
        #   transpose：attn = attn.transpose(0, 3, 6, 1, 4, 2, 5) = (b, h, d, hh, ws1, ww, ws2)
        #   reshape:   attn = qkv.reshape(b, (h*d), (hh*ws1), (ws*ws2))
        #
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
                         h=self.num_heads, d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws,
                         ws1=self.ws, ws2=self.ws)

        # cross-shaped window context interaction
        attn = attn[:, :, :H, :W]

        # attn_x: pad last dim by (0, 0) and 2nd to last by (0, 1)
        # attn_y: pad last dim by (0, 1) and 2nd to last by (0, 0)
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + self.attn_y(
            F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))   # global context

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


# Global-local transformer block (GLTB - ref. Fig. 2)
class GLTB(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# Feature maps generated by each stage of ResNet are fused with the corresponding feature maps
# of the decoder by a 1 × 1 convolution with the channel dimension in 64, i.e.,
# the skip connection. Specifically, the semantic features produced by the Resblocks
# are aggregated with the features generated by the GLTB of the decoder
# using a weighted sum operation.
class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(
            decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)

        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, decode_channels=64):
        super().__init__()

        # spatial path  utilizes a depth-wise convolution to produce a spatial-wise attentional map
        # S ∈ R (h×w×1), where h and w represent the spatial resolution of the feature map.
        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())

        # channel path employs a global average pooling layer to generate a channel-wise attentional
        # map C ∈ R1×1×c, where c denotes the channel dimension. The reduce & expand operation contains
        # two 1 × 1 convolutional layers, which first reduces the channel dimension c by a factor of 4
        # and then expands it to the original.
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),    # Global Average Pool
                                Conv(decode_channels, decode_channels //
                                     16, kernel_size=1),    # Reduce
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels,
                                     kernel_size=1),    # Expand
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(
            decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x

        # The attentional features generated by the two paths are further fused using a sum operation.
        x = pa + ca

        # A post-processing 1 × 1 convolutional layer and an upsampling operation are applied to produce the final segmentation map.
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


# Extra auxiliary head (AH block)
# The auxiliary head takes the fused feature of the 3 global–local Transformer blocks
# as the input and constructs a 3 × 3 convolution layer with batch normalization and ReLU,
# a 1 × 1 convolution layer and an upsampling operation to generate the output.
class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        # 3×3 convolution layer with batch normalization and ReLU
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes,
                             kernel_size=1)   # 1×1 convolution layer

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear',
                             align_corners=False)   # upsampling operation

        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(
            encoder_channels[-1], decode_channels, kernel_size=1)

        # GLTB, number of heads h are both set to 8
        self.glbt3 = GLTB(dim=decode_channels, num_heads=8,
                          window_size=window_size)

        self.ws3 = WS(encoder_channels[-2], decode_channels)     # weight sum

        self.glbt2 = GLTB(dim=decode_channels, num_heads=8,
                          window_size=window_size)

        self.ws2 = WS(encoder_channels[-3], decode_channels)

        self.glbt1 = GLTB(dim=decode_channels, num_heads=8,
                          window_size=window_size)

        self.ws1 = WS(encoder_channels[-4], decode_channels)

        self.frh = FeatureRefinementHead(decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.glbt3(self.pre_conv(res4))
        x = self.ws3(x, res3)

        x = self.glbt2(x)
        x = self.ws2(x, res2)

        x = self.glbt1(x)
        x = self.ws1(x, res1)

        x = self.frh(x)
        x = self.segmentation_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear',
                          align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@MODELS.register_module()
class UnetformerHead(BaseDecodeHead):
    def __init__(self, window_size=8, **kwargs):
        super(UnetformerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        encoder_channels = self.in_channels
        decode_channels = self.channels
        dropout = self.dropout_ratio

        self.decoder = Decoder(
            encoder_channels, decode_channels, dropout, window_size)

    def forward(self, x):
        h, w = x[0].size()[-2:]

        inputs = self._transform_inputs(x)
        x = self.decoder(inputs[0], inputs[1], inputs[2], inputs[3], h, w)
        x = self.cls_seg(x)
        return x
