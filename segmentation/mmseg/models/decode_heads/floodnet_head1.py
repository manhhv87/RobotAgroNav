# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath

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


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # Firstly, the channel dimension is increased by 1 * 1 convolution,
        # and the number of channels to be trained is fixed at the early stage of Bottleneck.
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, groups=4, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Then, 3 * 3 convolution is used to realize the first feature extraction.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Next, 1 * 1 convolution is employed again to achieve feature information fusion with the same number of channels.
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, groups=4, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        # Finally, 3 * 3 and 1 * 1 convolutions are performed respectively to realize feature re-extraction and information re-fusion.
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, groups=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

        self.conv5 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, groups=4, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Firstly, the channel dimension is increased by 1 * 1 convolution,
        # and the number of channels to be trained is fixed at the early stage of Bottleneck.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        residual1 = out

        # Then, 3 * 3 convolution is used to realize the first feature extraction.
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        residual2 = out

        # Next, 1 * 1 convolution is employed again to achieve feature information fusion with the same number of channels.
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = out + residual1

        # Finally, 3 * 3 convolution is performed to realize feature re-extraction.
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = out + residual2

        # Finally, 1 * 1 convolution is used to realize information re-fusion.
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        return out


# Global-local
class GlobalLocal(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        # local branch
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)

        # global branch
        self.glb = Bottleneck(dim, dim)

        self.proj = SeparableConvBN(dim, dim, kernel_size=3)

    def forward(self, x):
        """
        Args:
             x: input features with shape of (B, H, W, C)
        """
        B, C, H, W = x.shape

        # local context
        local = self.local2(x) + self.local1(x)

        # global context
        glb = self.glb(x)

        out = glb + local
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


# Global-local transformer block (GLTB - ref. Fig. 2)
class GLTB(nn.Module):
    def __init__(self, dim=256, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocal(dim)

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
        # Neu FloodNet_ViT thi comment, neu ko thi bo
        # x = F.interpolate(x, scale_factor=2, mode='bilinear',
        #                   align_corners=False)
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


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1):
        super(Decoder, self).__init__()

        # self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.pre_conv = Bottleneck(encoder_channels[-1], decode_channels)

        # GLTB, number of heads h are both set to 8
        self.glbt3 = GLTB(dim=decode_channels)

        self.ws3 = WS(encoder_channels[-2], decode_channels)     # weight sum

        self.glbt2 = GLTB(dim=decode_channels)

        self.ws2 = WS(encoder_channels[-3], decode_channels)

        self.glbt1 = GLTB(dim=decode_channels)

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

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@MODELS.register_module()
class UnetfloodnetHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(UnetfloodnetHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        encoder_channels = self.in_channels
        decode_channels = self.channels
        dropout = self.dropout_ratio

        self.decoder = Decoder(encoder_channels, decode_channels, dropout)

    def forward(self, x):
        h, w = x[0].size()[-2:]

        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(x)

        x = self.decoder(inputs[0], inputs[1], inputs[2], inputs[3], h, w)
        x = self.cls_seg(x)
        return x
