_base_ = [
    '../_base_/models/lmaswin.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='https://drive.usercontent.google.com/download?id=1tHNxQUffwNIfWFDKa4ql1klKGjCnbhwv&export=download&authuser=0&confirm=t&uuid=3f0fc36c-9936-4d25-9187-33b43954afe5&at=APZUnTXkSlUg7GRvllIndBNc0acn:1700898173032'

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='LMASwin',
        encoder_channels=(128, 256, 512, 1024),
        patch_size=2,
        atrous_rates=(6, 12),
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=8,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),

decode_head=dict(
        type='myFCNHead',
        in_channels=96,
        channels=96,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)])
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=None)

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