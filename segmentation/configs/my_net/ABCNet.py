# ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of Fine-Resolution remotely sensed imagery
_base_ = [
    '../_base_/models/abcnet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='open-mmlab://resnet18_v1c'

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        context_channels=(128, 256, 512),
        spatial_channels=(64, 64, 64, 128),
        out_channels=256,
        backbone_cfg=dict(
            type='ResNet', 
            depth=18,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))),

    decode_head=dict(
        type='myFCNHead',
        in_channels=256, 
        channels=256, 
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)])
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
