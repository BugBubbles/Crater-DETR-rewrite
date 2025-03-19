param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=2000,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            10,20,30
        ],
        type='MultiStepLR'),
]

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1)), norm_decay_mult=0.0),
    type='OptimWrapper')


train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
