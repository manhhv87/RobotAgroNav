# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='SmaAt_UNet',        
        init_cfg=dict(
          type='Pretrained', 
          checkpoint='https://objects.githubusercontent.com/github-production-release-asset-2e65be/382210636/c1c00b47-20af-4b20-9d23-ae74f0f61aab?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231117%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231117T153058Z&X-Amz-Expires=300&X-Amz-Signature=a32a018c6933871e02aabfcf74d8b45389e4f7d23fab0877b66f9e04b01553af&X-Amz-SignedHeaders=host&actor_id=67886698&key_id=0&repo_id=382210636&response-content-disposition=attachment%3B%20filename%3Dcswin_tiny_224.pth&response-content-type=application%2Foctet-stream')),

    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)]),


    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=384,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
