import matplotlib.pyplot as plt
import numpy as np
import mmcv
import pudb
import glob as glob
from agronav_mobilenetv3 import *
from mmseg.core import get_classes
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmseg.core.evaluation import get_palette
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# from segmentation.agronav_hrnet import *
# from segmentation.agronav_resnest import *


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--image',
        default='data/demo/GOPR0016.JPG',
        help='Image file')
    parser.add_argument(
        '-w', '--checkpoint',
        default='checkpoint/MobileNetV3.pth',
        help='weight file name'
    )
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--model',
        default='MobileNetV3',
        help='Model name')
    args = parser.parse_args()

    # if args.model == 'ResNest':
    #     cfg = cfg_resnest
    # elif args.model == 'HRNet':
    #     cfg = cfg_hrnet
    # else:
    #     cfg = cfg_mobilenetv3

    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint=None, device=args.device)
    cfg.load_from = args.checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    image_path = args.image
    save_dir = 'demo'
    image = mmcv.imread(image_path)
    result = inference_segmentor(model, image)

    save_name = f"{image_path.split(os.path.sep)[-1].split('.')[0]}"

    show_result_pyplot(model,
                       image,
                       result,
                       palette,
                       opacity=args.opacity,
                       out_file=f"{save_dir}/{save_name}.jpg")


if __name__ == '__main__':
    main()
