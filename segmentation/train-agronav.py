from agronav import *
from mmcv.cnn.utils import revert_sync_batchnorm
import mmcv
import os.path as osp
from mmseg.apis import train_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES
model.PALETTE = datasets[0].PALETTE
model = revert_sync_batchnorm(model)

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())
