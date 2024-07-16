_base_ = [
    '../_base_/models/dcswin.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://drive.usercontent.google.com/download?id=1jGgAbi15WLFUCRKNT0iBJjhIjeBTNAic&export=download&authuser=0&confirm=t&uuid=5fbbf2e9-09e7-4c87-8d96-e2d94f39eb8d&at=APZUnTUwJEHXVgEGYuOot4Lco-tO:1700896660304'
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),

    decode_head=dict(
        type='DCSwinHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=96,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)]
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.)}))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=8000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=8000,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=8, num_workers=2)
val_dataloader = dict(batch_size=8, num_workers=2)
test_dataloader = val_dataloader