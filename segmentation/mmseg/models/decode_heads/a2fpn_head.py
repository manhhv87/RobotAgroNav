# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(
            in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places,
                               out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(
            in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / \
            (width * height + torch.einsum("bnc, bc->bn",
             Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5, s4, s3, s2):
        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2,
                              mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(
            skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels,
                          upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(
                    out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels=(64, 128, 256, 512),
            pyramid_channels=64,
            segmentation_channels=64,
            dropout=0.2):
        super(Decoder, self).__init__()

        self.pre_conv = nn.Conv2d(
            encoder_channels[-1], pyramid_channels, kernel_size=1)

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[-2])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[-3])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[-4])

        self.s5 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=0)

        self.attention = AttentionAggregationModule(
            segmentation_channels * 4, segmentation_channels * 4)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        p5 = self.pre_conv(res4)
        p4 = self.p4([p5, res3])
        p3 = self.p3([p4, res2])
        p2 = self.p2([p3, res1])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        out = self.dropout(self.attention(s5, s4, s3, s2))
        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@MODELS.register_module()
class A2FPN(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(A2FPN, self).__init__(
            input_transform='multiple_select', **kwargs)

        encoder_channels = self.in_channels
        dropout = self.dropout_ratio

        self.decoder = Decoder(encoder_channels=encoder_channels, dropout=dropout)

    def forward(self, x):
        h, w = x[0].size()[-2:]

        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(x)
        x = self.decoder(inputs[0], inputs[1], inputs[2], inputs[3], h, w)    
        x = self.cls_seg(x)        
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x
