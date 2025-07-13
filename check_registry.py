# 在脚本开头添加导入
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))  # 确保能找到SDT目录

# 显式导入自定义模块
from .sdeformer import SDEFormer  # 关键导入

# 原有代码
from mmdet.registry import MODELS
print(f"已注册模型: {list(MODELS.module_dict.keys())}")
from mmdet.utils import register_all_modules  #我加进去的

# 显式导入SDEFormer以确保注册
from mmdet.models.backbones.sdeformer import SDEFormer  # 关键导入

from mmdet.models.backbones.cspnext import SDEFormer  # 关键导入

# 注册所有模块，包括自定义模块
register_all_modules()

from mmdet.registry import MODELS
print('SDEFormer' in MODELS.module_dict)  # 应返回 True
print('CSPNeXt' in MODELS.module_dict)  # 应返回 True
print(f"已注册模型: {list(MODELS.module_dict.keys())}")

# import mmengine

# # 替换为你的预测结果文件路径
# prediction_path = 'output_pkl/3-1.pkl'
# results = mmengine.load(prediction_path)

# # 检查第一个结果是否包含 'gt_instances' 键
# if 'gt_instances' not in results[0]:
#     print("预测结果文件中不包含 'gt_instances' 键。")
# else:
#     print("预测结果文件中包含 'gt_instances' 键。")