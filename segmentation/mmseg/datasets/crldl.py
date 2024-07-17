# https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html
# https://webcache.googleusercontent.com/search?q=cache:https://mducducd33.medium.com/sematic-segmentation-using-mmsegmentation-bcf58fb22e42
# https://github.com/DequanWang/actnn-mmseg/tree/icml21/docs/tutorials

# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CropRowDataset(BaseSegDataset):
    """CRDLD dataset.

    In segmentation map annotation for CRDLD dataset, 0 stands for background, which
    is not included in 2 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=("crop", "background"),
        palette = [[1], [0]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            reduce_zero_label=False,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
