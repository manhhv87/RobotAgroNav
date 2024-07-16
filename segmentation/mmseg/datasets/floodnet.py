# https://mmsegmentation.readthedocs.io/en/main/advanced_guides/add_datasets.html
# https://webcache.googleusercontent.com/search?q=cache:https://mducducd33.medium.com/sematic-segmentation-using-mmsegmentation-bcf58fb22e42
# https://github.com/DequanWang/actnn-mmseg/tree/icml21/docs/tutorials

# REFUGE dataset

# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FloodNetDataset(BaseSegDataset):
    """FloodNet dataset.

    In segmentation map annotation for FloodNet, 0 stands for background, which
    is not included in 2 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', 'building-flooded', 'building-nonflooded', 'road-flooded',
                 'road-nonflooded', 'water', 'tree', 'vehicle', 'pool', 'grass'),
        # palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 120], [0, 0, 255], [255, 0, 255], 
        #          [70, 70, 220], [102, 102, 156], [190, 153, 153], [180, 165, 180]])
        palette=[[68,1,84], [71,39,119], [62,73,137], [48,103,141], [37,130,142], [30,157,136], 
                 [53,183,120], [109,206,88], [181,221,43], [253,231,36]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
