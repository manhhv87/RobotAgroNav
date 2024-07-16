# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation *
                               (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )


class SharedSpatialAttention(nn.Module):
    def __init__(self, in_places, eps=1e-6):
        super(SharedSpatialAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(
            in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        # print('q', Q.shape)
        K = self.l2_norm(K)
        # print('k', K.shape)

        tailor_sum = 1 / \
            (width * height + torch.einsum("bnc, bc->bn",
             Q, torch.sum(K, dim=-1) + self.eps))
        # print('tailor_sum', tailor_sum.shape)
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        # print('value_sum', value_sum.shape)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        # print('value_sum', value_sum.shape)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        # print('matrix',matrix.shape)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        # print('matrix_sum', matrix_sum.shape)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        # print('weight_value', weight_value.shape)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class SharedChannelAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(SharedChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = x.view(batch_size, chnnels, -1)
        K = x.view(batch_size, chnnels, -1)
        V = x.view(batch_size, chnnels, -1)

        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)

        tailor_sum = 1 / \
            (width * height + torch.einsum("bnc, bn->bc",
             K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)

        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class DownConnection(nn.Module):
    def __init__(self, inplanes, planes, stride=2):
        super(DownConnection, self).__init__()
        self.convbn1 = ConvBN(inplanes, planes, kernel_size=3, stride=1)
        self.convbn2 = ConvBN(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ConvBN(inplanes, planes, stride=stride)

    def forward(self, x):
        residual = x
        x = self.convbn1(x)
        x = self.relu(x)
        x = self.convbn2(x)
        x = x + self.downsample(residual)
        x = self.relu(x)

        return x


class DCFAM(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12)):
        super(DCFAM, self).__init__()

        rate_1, rate_2 = tuple(atrous_rates)
        self.conv4 = Conv(
            encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.conv1 = Conv(
            encoder_channels[0], encoder_channels[0], kernel_size=1)
        self.lf4 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-1], encoder_channels[-2], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(
                                     encoder_channels[-2], encoder_channels[-3], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.lf3 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(
                                     encoder_channels[-3], encoder_channels[-4], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))

        self.ca = SharedChannelAttention()
        self.pa = SharedSpatialAttention(in_places=encoder_channels[2])
        self.down12 = DownConnection(encoder_channels[0], encoder_channels[1])
        self.down231 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down232 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down34 = DownConnection(encoder_channels[2], encoder_channels[3])

    def forward(self, x1, x2, x3, x4):
        out4 = self.conv4(x4) + self.down34(self.pa(self.down232(x2)))
        out3 = self.pa(x3) + self.down231(self.ca(self.down12(x1)))
        out2 = self.ca(x2) + self.lf4(out4)
        del out4
        out1 = self.lf3(out3) + self.conv1(x1)
        del out3

        return out1, out2


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05,
                 atrous_rates=(6, 12)):
        super(Decoder, self).__init__()

        self.dcfam = DCFAM(encoder_channels, atrous_rates)
        self.up = nn.Sequential(
            ConvBNReLU(encoder_channels[1], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.conv = ConvBNReLU(encoder_channels[0], encoder_channels[0])
        self.init_weight()

    def forward(self, x1, x2, x3, x4):
        out1, out2 = self.dcfam(x1, x2, x3, x4)
        x = out1 + self.up(out2)
        x = self.dropout(x)
        x = self.conv(x)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@MODELS.register_module()
class DCSwinHead(BaseDecodeHead):
    def __init__(self, atrous_rates=(6, 12), **kwargs):
        super(DCSwinHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        encoder_channels = self.in_channels
        dropout = self.dropout_ratio

        self.decoder = Decoder(encoder_channels, dropout, atrous_rates)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(x)

        x = self.decoder(inputs[0], inputs[1], inputs[2], inputs[3])
        x = self.cls_seg(x)
        output = self.up(x)

        return output
