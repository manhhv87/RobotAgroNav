# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class ClsHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(ClsHead, self).__init__(**kwargs)

    def forward(self, inputs):        
        """Forward function."""        
        h, w = inputs.size()[-2:]   

        output = self.cls_seg(inputs)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)   
        return output
