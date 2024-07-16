# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .boundary_loss import BoundaryLoss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .huasdorff_distance_loss import HuasdorffDisstanceLoss
from .lovasz_loss import LovaszLoss
from .ohem_cross_entropy_loss import OhemCrossEntropy
from .silog_loss import SiLogLoss
from .tversky_loss import TverskyLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .my_loss import My_CrossEntropyLoss
from .cross_entropy_lossv1 import CrossEntropyLossV1
from .my_dice_loss import MultiClassDiceLoss
from .my_boundary_loss import MultiClassBoundaryLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'TverskyLoss', 'OhemCrossEntropy', 'BoundaryLoss',
    'HuasdorffDisstanceLoss', 'SiLogLoss', 'My_CrossEntropyLoss',
    'CrossEntropyLossV1', 'MultiClassDiceLoss','MultiClassBoundaryLoss'
]
