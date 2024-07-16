_base_ = [
    '../_base_/models/mtunet.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# checkpoint='https://doc-0k-64-docs.googleusercontent.com/docs/securesc/tvoi3fua39rd37j1lobniftt2hb7dg0u/o62hbc4cq10gvh9p8juso13f908j8e6d/1701101700000/15959628209787595289/09412009828125765653/1eo6d-d_kR0qbHBIHq49TQ1CFpPLypJUT?e=download&ax=AEqgLxlvLbUrD07Hirgw1U5HgSShGQHX-fUSUJFdeRiVA0VsSqSZNeO7kTMwFY1QHPgy1-0y_MV84s_DAXqO_oUsUyxi6n2jAGWC6I3sqvthv4c_kmwb2bxsqwU9sTouXnlEY-tXbq3XIiCmbm7qvQQz57kxTrtFqp_-u_zScYXgeprzFfrBclrKkGM050tVRP7FXqui6a8ov_JKu9D0PfFgxymqcUogjCHsB2HkPH1EiNOrSTgEN9wQYLTLcYaCiCwIVr4UrZxYnbnAiV5UWKOhqHVVilZXAD8yLPLfhGe_vOOb5Zts8xyFMlzZlyoja1fPOvYFMJWlEglE-vpAFB6OgHDegG5WtusNuKDhEJVpS6qIx3Eg1IugCBQwLAru3_UQi-DQlAvCfUWx9H6J0Tq71Ftxv-LAC0SSIruhk4gkFPdIoTbXgow5fueWDVbNBEOFDFXuwPFxXWwvRu9ayuI3o4C0mhfRPmKbV0akLyOlghQ8o96pHR3d0dCOT-8kWURVWUKmpTBpyUkOAPlJTuaGi2J0YyLshHDoKfy4IDeDTf07ZjvGR7KPaCqoAzYF6zSbwGL4rY7YNYgy7ZdTl3KalwBz1I1yeHBK_PsHxDfdb0uUW-GML7k0f5pyEgR8Io3G7H4c1Wstqg8fAg96o8Vykp5laWTVOywS-aWJSLI0v3Q2cDwjbo1-4NrpG_EU5D53HB6yJaEk81m1cmmJjfCfZsy5ulIOap0-LbskzEp6u0OXwh33z_aA6mhDD7zPX-l7ijo4wQjgxifJoRIjKu9r0V0dwUzsMT4AATPcBFclf6G9hiKcEixF0K6GGURDrbaqbM7CjUOm4uUTQCgCyO_uTyakyUniN_zYfniawo88nCZxrOom_ljD-nxxmbcIYyOWnN4WuRMZtBeHu2VPfmG_lfaJzGyvUeDyYtNt7BAtgXpdd5vPLaIjrrSU_Ggd31d73rmy3A-4734NVNuSUJhBUzddnBChSsySbG0C6dPM15HlKJXH2T6BP4iLjX4kHqV5afX5ovEPSPggEi7z3dEWWTczWftLAlE&uuid=7bafc451-882c-40c5-8f9e-cf476f8c5f87&authuser=0&nonce=8ccrem7m24gnq&user=09412009828125765653&hash=0mjur0k2a6qdh95s5mie1aqo1k7fn7tf'

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='MTUNet',
        encoder=[256, 512],
        decoder=[1024, 512],
        bottleneck=1024,
        num_heads=8,
        win_size=4,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),

    decode_head=dict(
        type='ClsHead',
        in_channels=64,
        in_index=0,
        channels=64,
        num_classes=10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.3),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.7)]
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))

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
