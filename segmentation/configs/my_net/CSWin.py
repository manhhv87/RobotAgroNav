_base_ = [
    '../_base_/models/cswin.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint='https://objects.githubusercontent.com/github-production-release-asset-2e65be/382210636/87c0205e-d6c5-4055-9ce1-adb096034161?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231125%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231125T134716Z&X-Amz-Expires=300&X-Amz-Signature=ccc1c8bb94cfa9a907dd96f5717c9fb7456ccee97043ce2de962c6122d00dffe&X-Amz-SignedHeaders=host&actor_id=67886698&key_id=0&repo_id=382210636&response-content-disposition=attachment%3B%20filename%3Dcswin_base_224.pth&response-content-type=application%2Foctet-stream'

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='CSWin',
        embed_dim=96,
        depth=[2,4,32,2],
        num_heads=[4,8,16,32],
        split_size=[1,2,7,7],
        drop_path_rate=0.6,
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),

    decode_head=dict(
        type='myFCNHead',
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
