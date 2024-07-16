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
    pretrained=None,

    backbone=dict(
        type='DBUNet',
        img_size=224,
        in_channels=3,
        out_indices=(0, 1, 2, 3, 4, 5, 6, 7),
        patch_size=28,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        init_cfg=None),

    decode_head=dict(
        type='DBUNetHead',
        in_channels=(256, 512, 1024, 2048, 65, 65, 65, 65), 
        in_index=(0, 1, 2, 3, 4, 5, 6, 7),
        channels=32,
        dropout_ratio=0.2,
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
