_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/crdld.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
        out_channels=2,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=True,
                loss_weight=0.3,
                avg_non_ignore=False),
            dict(
                type='DiceLoss', 
                loss_name='loss_dice', 
                loss_weight=0.7)]))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=None
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
