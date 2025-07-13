default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每1个epoch保存一次
        max_keep_ckpts=3,  # 最多保留3个最新的检查点
        save_best='coco/bbox_mAP_50',  # 保存验证集上性能最佳的模型
        rule='greater' # 指标越大越好
        # save_optimizer=False  # 不保存优化器状态，减小文件体积，保存优化器状态：可以从任意检查点完全恢复训练，包括学习率、动量等参数的当前值。
        # 不保存优化器状态：只能加载模型权重用于推理或微调，无法恢复训练过程（例如无法继续之前的学习率调度）。
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
# load_from ="/lxh/spike-driven-transformer/mmdet3/work_dirs/t1_adamw_0.440_15m.pth"
load_from =None
# resume = "/lxh/spike-driven-transformer/mmdet3/work_dirs/t1_adamw_resume_0.466_55m_epoch_12.pth"
resume = None