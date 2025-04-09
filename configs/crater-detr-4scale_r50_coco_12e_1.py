_base_ = ['./datasets/craters_1.py', 
          './schedular/200epochs.py', 
          './default_runtime.py']
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects',
    ])

model = dict(
    type='CraterDETR',
    as_two_stage=True,
    num_queries=900,
    with_box_refine=True,
    aux_start_level=0,
    aux_end_level=3,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(1, 2, 3),
        style='pytorch',
        type='ResNet'),
    neck=dict(
        act_cfg=None,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    fpn=dict(
        in_channels=256,
        out_channels=256,
        num_levels=4,
        ksize=3,
        stride=2,
        type='CraterDETRFPN',
    ),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='SOSIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='SOSDINOHead'),
    aux_bbox_head=dict(
        type="AACLAYOLOV3Head",
        num_classes=1,
        in_channels=[256, 256, 256],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
          type='YOLOAnchorGenerator',
          base_sizes=[[(116, 90), (156, 198), (373, 326)],
                      [(30, 61), (62, 45), (59, 119)],
                      [(10, 13), (16, 30), (33, 23)]],
          strides=[64, 32, 32]
          ),
          featmap_strides=[64, 32, 32]),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675,116.28,103.53],
        pad_size_divisor=1,
        std=[58.395,57.12,57.375],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5,
        type="AdnQueryGenerator"),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        bbox_head=dict(
          assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
        aux_bbox_head=dict(
          assigner=dict(
                type='GridAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.3),
          aux_cfg=dict(
                dict(
                  nms=dict(iou_threshold=0.5, type='nms'),
                  nms_pre=1000,
                  max_per_img=300,
                  conf_thr=0.4)
                  )
            )
        )
)

work_dir = './logs/crater-detr-4scale_r50_coco_30e'
