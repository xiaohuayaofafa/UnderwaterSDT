_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/uod_instance.py',#修改为我的 udd配置文件，同时修改下面的 roi_head 里的类别数量为 3
    '../_base_/schedules/schedule_1x.py',
     '../_base_/default_runtime.py'  #在这里修改运行时保存 epoch 的次数，只保存最佳 epoch 等
]

pretrained = "V3_19.0M_1x4.pth"
#先注释掉，不使用预训练的权重 

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    # 我把 mask 改为 false
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), 
    dict(
        type='Resize', 
        scale=(853, 512), 
        keep_ratio=True),#我改成 640x480 原来是 853,512
    dict(type='RandomFlip', prob=0.5),

    dict(
        type='RandomChoice',
        #随机选择数据增强操作，定义了两种
        # 每次处理一张图像时都会进行随机选择，因此在同一个 epoch 中，不同的图像可能会使用不同的数据增强方法。这增加了训练数据的多样性，有助于模型学习到更丰富的特征，提高模型的泛化能力。
       transforms=[
            [
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    # 原来是 4 我改成 2 了 爆显存了试一下
    batch_size=4,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SDEFormer',
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)#先注释掉不知道是不是跟预训练权重相关的
    ),
    neck=dict(
        type='SpikeFPN',
        in_channels=[64, 128, 256, 360],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='SpikeRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # 这里我做了修改，为了适应水下数据集 调整锚点尺寸，可能需要减小尺度
            scales=[4,8,16],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
        #                loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # loss_bbox=dict(type='GIoULoss', loss_weight=1.0)
    ),
    roi_head=dict(
        # 这里调用了SpikeStandardRoIHead，而这个头的定义在 spike_standar_roi_head 中，这里面包含 mask 的内容导致运行问题
        type='SpikeStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='SharedSpike2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,  #修改为uod 数据集为 4 个类别
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
            #                loss_weight=2.0),
            # loss_bbox=dict(type='GIoULoss', loss_weight=1.0), #有bug，loss变负数
        ),

        # 我的 urpc 数据集没有掩码 所以先注释掉这一部分，好像也可以通过在一开始把模型头设置为 fastrcnn 来解决，因为 maskrcnn 需要掩码
        # mask_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        # mask_head=dict(
        #     type='SpikeFCNMaskHead',
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=80,
        #     loss_mask=dict(
        #         type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ),)

max_epochs = 30
max_iter = max_epochs * 23454
train_cfg = dict(max_epochs=max_epochs)

# # learning rate 这里设置了学习率，因为是引用这个配置文件的参数跑的 
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]


# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             # 'absolute_pos_embed': dict(decay_mult=0.),
#             # 'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0002,
#         eps=1e-8,
#         betas=(0.9, 0.999),
#         weight_decay=0.05),
# )

# fp16 = dict(loss_scale=512.)

# 结合 batchsize=16 所做出的修改
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001,
        by_epoch=False, 
        begin=0,
        end=2000  #延长预热时间
        ),
    dict(
        type='CosineAnnealingLR',
        T_max=98080,  # 总迭代100080 - 预热2000 = 98080
        eta_min=1e-6,  # 最小学习率
        begin=2000,    # 从第2000次迭代开始
        end=100080,    # 到第100080次迭代结束
        by_epoch=False
        # type='MultiStepLR',
        # begin=0,
        # end=max_epochs,
        # by_epoch=True,
        # milestones=[8, 11],
        # gamma=0.1
        )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0, norm_type=2),  # 添加梯度裁剪
    accumulative_counts=8,  # 可选，模拟batch_size=16
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,  #学习率增大四倍
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0.05),
)
# fp16 = dict(loss_scale=512.)
