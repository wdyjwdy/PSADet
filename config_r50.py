_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py'
model = dict(
    bbox_head=dict(
        loss_bbox_init=dict(_delete_=True, type='DIoULoss', loss_weight=0.5),
        loss_bbox_refine=dict(_delete_=True, type='DIoULoss', loss_weight=1.0)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ])

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4 * 0.5 * 0.25, weight_decay=0.1)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

# augmentation strategy originates from DETR.
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[[
             dict(type='Resize',
                  img_scale=[
                      (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                      (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                      (736, 1333), (768, 1333), (800, 1333)
                  ],
                  multiscale_mode='value',
                  keep_ratio=True)
         ],
                   [
                       dict(type='Resize',
                            img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True),
                       dict(type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                       dict(type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                       (576, 1333), (608, 1333), (640, 1333),
                                       (672, 1333), (704, 1333), (736, 1333),
                                       (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True)
                   ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline))
lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)