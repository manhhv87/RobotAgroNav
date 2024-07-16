# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class GeneralHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(GeneralHead, self).__init__(**kwargs)
        assert self.in_channels == self.channels
        self.convs = nn.Identity()

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        return feats
    
    def forward(self, inputs):
        """Forward function."""
        if len(inputs) == 3:
            output = self.cls_seg(inputs)
        else:
            output = self._forward_feature(inputs)

        output = self.cls_seg(inputs)
        return output
