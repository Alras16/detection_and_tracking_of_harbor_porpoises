checkpoint_config = dict(interval=20)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
evaluation = dict(interval=5, metric='accuracy')
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True,
    step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
model = dict(
    type='ImageClassifier',
    backbone=dict(type='CRNetv2'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'CustomDataset'
classes = [
    'n01440764', 'n01496331', 'n01537544', 'n01608432', 'n01632777',
    'n01667778', 'n01693334', 'n01729977', 'n01742172', 'n01756291',
    'n01773797', 'n01795545', 'n01819313', 'n01833805', 'n01877812',
    'n01930112', 'n01968897', 'n01986214', 'n02012849', 'n02028035',
    'n01443537', 'n01498041', 'n01560419', 'n01614925', 'n01644900',
    'n01675722', 'n01695060', 'n01734418', 'n01749939', 'n01770081',
    'n01774384', 'n01796340', 'n01820546', 'n01843383', 'n01883070',
    'n01943899', 'n01978287', 'n02002556', 'n02013706', 'n02037110',
    'n01484850', 'n01514668', 'n01582220', 'n01622779', 'n01664065',
    'n01677366', 'n01698640', 'n01735189', 'n01751748', 'n01770393',
    'n01774750', 'n01798484', 'n01824575', 'n01847000', 'n01910747',
    'n01944390', 'n01978455', 'n02006656', 'n02018207', 'n02051845',
    'n01491361', 'n01514859', 'n01592084', 'n01630670', 'n01665541',
    'n01685808', 'n01728572', 'n01739381', 'n01753488', 'n01773157',
    'n01775062', 'n01806143', 'n01828970', 'n01855672', 'n01914609',
    'n01950731', 'n01984695', 'n02007558', 'n02018795', 'n02058221',
    'n01494475', 'n01531178', 'n01601694', 'n01632458', 'n01667114',
    'n01687978', 'n01729322', 'n01740131', 'n01755581', 'n01773549',
    'n01776313', 'n01818515', 'n01829413', 'n01860187', 'n01924916',
    'n01955084', 'n01985128', 'n02011460', 'n02027492', 'n02077923'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(288, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_prefix='data/imagenet-100/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='data/imagenet-100/validate',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(288, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        data_prefix='data/imagenet-100/validate',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=(288, -1),
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
work_dir = './work_dirs/crnet_8xb32_in100'
gpu_ids = [0]
