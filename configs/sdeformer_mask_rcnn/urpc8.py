#从 _base_ 目录下引用了四个基础配置文件，分别是 Mask R-CNN 模型的基础配置、
#COCO 数据集实例分割任务的配置、训练调度的配置以及默认运行时的配置。
#这些基础配置提供了一些通用的设置，本配置文件可以在此基础上进行修改和扩展。
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/urpc_instance.py',#修改为我的 urpc 配置文件，同时修改下面的 roi_head 里的类别数量为 4
    '../_base_/schedules/schedule_1x.py',
     '../_base_/default_runtime.py'
]

#预训练权重
pretrained = "V3_19.0M_1x4.pth"

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), # 我把 mask 改为 false
    dict(type='Resize', scale=(640,480), keep_ratio=True),#我改成 640x480 原来是 853,512
    dict(type='RandomFlip', prob=0.5),
  # MSR
    dict(
        type='MultiRetinex',
        model='MSR',  # 使用 MSR 模型
        sigma=[30, 150, 300],  # 高斯核的方差
        restore_factor=2.0,  # 控制颜色修复的非线性
        color_gain=10.0,  # 控制颜色修复增益
        gain=128.0,  # 图像像素值改变范围的增益
        offset=128.0  # 图像像素值改变范围的偏移量
    ),
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
    batch_size=8,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))

model = dict(
    #定义在mmdet/model/detectors
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='SpikeFPN',
        in_channels=[64, 128, 256, 360],
        out_channels=256,
        #从 backbone 输出的特征图只有 4 张，但 SpikeFPN 通过最大池化操作在最后一个输出特征图上生成了额外的特征图，
        #从而使得最终输出的特征图数量达到 5 张，这些特征图会被输入到 RPN 中进行后续的目标检测任务。
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
            num_classes=4,  #修改为urpc 数据集为 4 个类别
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

#  
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=300  # ≈0.64个epoch (150次迭代)
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=13830,  # 30 - 0.64 = 29.36个epoch
        eta_min=5e-7,  # 保持您设定的最小值
        by_epoch=False,
        begin=300,
        end=14130
    )
]
optim_wrapper = dict(
    type='OptimWrapper',
    # 梯度裁剪调整（相对宽松）
    clip_grad=dict(max_norm=6.0, norm_type=2),  # 原为3.0
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.6)  # 骨干网络学习率折扣
        }
    ),
    optimizer=dict( 
        _delete_=True,
        type='AdamW',
        lr=0.0045,
        eps=1e-8,
        betas=(0.9, 0.99),
        weight_decay=0.008
    )
)
# 混合精度设置
fp16 = dict(loss_scale=512.)


