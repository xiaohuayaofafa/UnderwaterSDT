_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/udd_instance.py',#修改为我的 udd配置文件，同时修改下面的 roi_head 里的类别数量为 3
    '../_base_/schedules/schedule_1x.py',
     '../_base_/default_runtime.py'
]

pretrained = "V3_19.0M_1x4.pth"
#先注释掉，不使用预训练的权重 

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    # 我把 mask 改为 false
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), 
    dict(type='Resize', scale=(640,480), keep_ratio=True),#我改成 640x480 原来是 853,512
    dict(type='RandomFlip', prob=0.5),

    # 我新增了三行为了 urpc 而做处理的数据 在MMDetection中，所有数据增强操作都需要先注册到TRANSFORMS注册表中
    # dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 颜色抖动
    # dict(type='RandomBlur', prob=0.3, radius=3),  # 随机模糊，模拟水下悬浮颗粒
    # dict(type='RandomGamma', gamma_range=(0.7, 1.5)),  # 调整亮度和对比度

    dict(
        type='RandomChoice',
        #随机选择数据增强操作，定义了两种
        # 每次处理一张图像时都会进行随机选择，因此在同一个 epoch 中，不同的图像可能会使用不同的数据增强方法。这增加了训练数据的多样性，有助于模型学习到更丰富的特征，提高模型的泛化能力。
        transforms=[
            [
            dict(
                type='RandomChoiceResize',
                # 为了适应数据集尺寸做的修改，避免太大
                scales=[(480, 640), (512, 640), (544, 640), (576, 640), (608, 640)],
                keep_ratio=True)
        ],
        # 如果选择了第二个列表，那么会依次对图像进行 RandomChoiceResize、RandomCrop 和 RandomChoiceResize 操作
                    [
                        dict(
                            type='RandomChoiceResize',
                            # 为了适应数据集做的修改
                            scales=[(400, 640), (480, 640), (560, 640)],
                            keep_ratio=True),
                        dict(
                            # type='RandomCrop',
                            # crop_type='absolute_range', 
                            # crop_size=(384, 288),
                            # allow_negative_crop=True
                            type='RandomCrop',
                            crop_type='relative_range',  # 相对比例 原来的代码裁剪完完全没有目标了所以进行修改
                            crop_size=(0.7, 0.7),  # 裁剪70%区域
                            allow_negative_crop=False),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 640), (512, 640), (544, 640), (576, 640), (608, 640)], # 调整为适合640×480的范围
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    # 原来是 4 我改成 2 了 爆显存了试一下
    batch_size=16,
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
            num_classes=3,  #修改为urpc 数据集为 3 个类别
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
    ),)

max_epochs = 30
max_iter = max_epochs * 23454
train_cfg = dict(max_epochs=max_epochs)

# 结合 batchsize=16 所做出的修改
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.01,
        by_epoch=False, 
        begin=0,
        end=300  #延长预热时间
        ),
    dict(
        type='CosineAnnealingLR',  # 改用余弦退火
        T_max=50,    # 90%的epoch使用余弦衰减
        eta_min=1e-6,              # 最小学习率
        by_epoch=True,
        begin=1,
        end=30
        )
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=4.0, norm_type=2),  # 添加梯度裁剪
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.5)  # 骨干网络学习率折扣
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0009,  #学习率增大四倍
        eps=1e-8,
        betas=(0.9, 0.99),
        weight_decay=0.008),
)
fp16 = dict(loss_scale=512.)

# #epoch=30
# param_scheduler = [
#     # 预热阶段（500次迭代）
#     dict(
#         type='LinearLR',
#         start_factor=0.001,  # 初始学习率=0.0008 * 0.001=8e-7
#         by_epoch=False,
#         begin=0,
#         end=500
#     ),
#     # 主训练阶段（余弦退火）
#     dict(
#         type='CosineAnnealingLR',
#         T_max=3070,  # 3570-500=3070
#         eta_min=1e-6,
#         by_epoch=False,  # 改为按迭代次数调整
#         begin=500,
#         end=3570
#     )
# ]
# optim_wrapper = dict(
#     type='OptimWrapper',
#     clip_grad=dict(max_norm=8.0, norm_type=2),  # 放宽梯度裁剪
#     paramwise_cfg=dict(
#         custom_keys={
#             'norm': dict(decay_mult=0.),
#             # 新增水下检测专用调整
#             'backbone': dict(lr_mult=0.7),  # 骨干网络降学习率
#             'neck': dict(lr_mult=1.0),
#             'head': dict(lr_mult=1.3)  # 检测头增强学习
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0008,  # 保持您设定的基础学习率
#         eps=1e-8,
#         betas=(0.9, 0.99),
#         weight_decay=0.01
#     )
# )
# fp16 = dict(loss_scale=512.)

#新的 epoch=30 的策略
# param_scheduler = [
#     # 预热阶段 (缩减比例以适应更长训练)
#     dict(
#         type='LinearLR',
#         start_factor=0.001,  # 初始学习率=0.001 * 0.001=1e-6
#         by_epoch=False,
#         begin=0,
#         end=600  # 原500 → 600 (10%延长，但比例下降)
#     ),
#     # 恒定期 (比例压缩)
#     dict(
#         type='ConstantLR',
#         factor=1.0,
#         by_epoch=False,
#         begin=600,
#         end=3000  # 1500 → 3000 (保持约总迭代42%不变)
#     ),
#     # 退火期 (等比例延长)
#     dict(
#         type='CosineAnnealingLR',
#         T_max=4110,  # 2070 → 4110 (7110-3000)
#         eta_min=1e-6,
#         by_epoch=False,
#         begin=3000,
#         end=7110
#     )
# ]
# optim_wrapper = dict(
#     type='AmpOptimWrapper',  # 使用混合精度封装
#     accumulative_counts=2,   # 梯度累积维持逻辑batch=16
#     loss_scale='dynamic',    # 自动调整混合精度缩放
    
#     # 优化器配置
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=0.0009,           # 微降至0.0009 (补偿batch变小)
#         eps=1e-8,
#         betas=(0.9, 0.99),
#         weight_decay=0.01
#     ),
#     clip_grad=dict(max_norm=6.0, norm_type=2),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone': dict(lr_mult=0.8),
#             'neck': dict(lr_mult=1.0),
#             'head': dict(lr_mult=1.2),
#             'norm': dict(decay_mult=0.)
#         }
#     )
# )
