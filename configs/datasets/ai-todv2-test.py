backend_args = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_pipeline = test_pipeline

metainfo = dict(
    classes=('airplane', 'bridge', 'storage-tank', 'ship',
              'swimming-pool', 'vehicle', 'person', 'wind-mill')
    )
data_root = '/home/temp/AI-TOD/aitod/'
dataset_type = 'CocoDataset'
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/aitodv2_val.json',
        backend_args=None,
        data_prefix=dict(img='images/val'),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=data_root + 'annotations/aitodv2_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/aitodv2_test.json',
        backend_args=None,
        data_prefix=dict(img='images/test'),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=data_root + 'annotations/aitodv2_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')


train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/aitodv2_train.json',
        backend_args=None,
        data_prefix=dict(img='images/train'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline,
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

custom_hooks = [
  dict(type='ValTestHook', loop=test_cfg, dataloader=test_dataloader, evaluator=test_evaluator),
]
