#这是我自己定义的文件，用于定义数据集的配置参数，路径等内容

# dataset settings
dataset_type = 'URPCDataset'
data_root = 'data/urpc/'  # 替换为 URPC 数据集的实际路径

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),#我调整了缩放尺寸，我不确定有什么用，只是为了保持和原始图片大小一致 250630
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation/train.json',  # 替换为 URPC 训练集的标注文件路径
        data_prefix=dict(img='train/'),  # 替换为 URPC 训练集的图像文件夹路径
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation/val.json',  # 替换为 URPC 验证集的标注文件路径
        data_prefix=dict(img='valid/'),  # 替换为 URPC 验证集的图像文件夹路径
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

# 新增独立测试集配置，删掉原来的test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation/test.json',  # 测试集标注
        data_prefix=dict(img='test/'),    # 测试集图像
        test_mode=True,
        pipeline=test_pipeline,  # 使用专用测试流程
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation/val.json',  # 替换为 URPC 验证集的标注文件路径
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
    

# 我新增的
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation/test.json',  # 测试集标注
    metric='bbox',
    format_only=False,  # 是否仅保存结果不评估
    # metric_options={
    #     'bbox': {'classwise': True},  # 启用bbox的类别AP计算，好像已经废弃了 换一种写法
    # },
    # dump_gt=True,#我改的，保存 gt 值，用不了，不兼容
    outfile_prefix='./auto_results/urpc_test',
    classwise=True,
    backend_args=backend_args)