import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# FFB module
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


@MODELS.register_module()
class DBUNetHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(DBUNetHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        features = self.in_channels

        self.bottleneck = Bottleneck(512, 1024)
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2 + 65 + feature, feature * 2, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.vitLayer_UpConv = nn.ConvTranspose2d(
            65, 65, kernel_size=2, stride=2)

        self.final_conv1 = nn.ConvTranspose2d(64, 32, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_4, x_8, x_16, x_32, vit_layerInfo_0, vit_layerInfo_1, vit_layerInfo_2, vit_layerInfo_3 = x

        vit_layerInfo = [vit_layerInfo_0, vit_layerInfo_1,
                         vit_layerInfo_2, vit_layerInfo_3]

        x = self.bottleneck(x_32)

        # Flip to positive order. 0 means the fourth layer...3 means the first layer
        vit_layerInfo = vit_layerInfo[::-1]

        v = vit_layerInfo[0].view(4, 65, 32, 32)
        x = torch.cat([x, x_32, v], dim=1)
        x = self.ups[0](x)
        x = self.ups[1](x)

        v = vit_layerInfo[1].view(4, 65, 32, 32)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, x_16, v], dim=1)
        x = self.ups[2](x)
        x = self.ups[3](x)

        v = vit_layerInfo[2].view(4, 65, 32, 32)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, x_8, v], dim=1)
        x = self.ups[4](x)
        x = self.ups[5](x)

        v = vit_layerInfo[3].view(4, 65, 32, 32)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        v = self.vitLayer_UpConv(v)
        x = torch.cat([x, x_4, v], dim=1)
        x = self.ups[6](x)
        x = self.ups[7](x)

        out1 = self.final_conv1(x)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.cls_seg(out)

        return out
