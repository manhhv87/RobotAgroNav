# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from .decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class LinearHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(LinearHead, self).__init__(**kwargs)
        num_classes = self.num_classes
        channels = self.channels

        self.head = nn.Linear(channels, num_classes)

    def forward(self, inputs):        
        """Forward function."""        
        output = self.head(inputs)
        return output
