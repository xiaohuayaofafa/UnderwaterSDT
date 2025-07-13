#这是我自己定义的文件，用于定义一个 UrpcDatasets 类

import copy
import torch
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class URPCDataset(BaseDetDataset):
    """Dataset for URPC."""

    # 根据 URPC 数据集的实际类别和调色板进行修改
    METAINFO = {
        'classes': ('echinus', 'holothurian', 'scallop','starfish'),  # 替换为 URPC 实际的类别
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]  # 调色板可自定义
    }

    # 假设 URPC 数据集的标注 ID 是唯一的
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            import json
            with open(local_path, 'r') as f:
                self.urpc = json.load(f)

        # 获取类别 ID 和类别到标签的映射
        self.cat_ids = []
        self.cat2label = {}
        for i, cat in enumerate(self.metainfo['classes']):
            for category in self.urpc.get('categories', []):
                if category['name'] == cat:
                    cat_id = category['id']
                    self.cat_ids.append(cat_id)
                    self.cat2label[cat_id] = i
                    break

        self.cat_img_map = {}
        for ann in self.urpc.get('annotations', []):
            cat_id = ann['category_id']
            img_id = ann['image_id']
            if cat_id not in self.cat_img_map:
                self.cat_img_map[cat_id] = []
            if img_id not in self.cat_img_map[cat_id]:
                self.cat_img_map[cat_id].append(img_id)

        img_ids = [img['id'] for img in self.urpc.get('images', [])]
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = next((img for img in self.urpc['images'] if img['id'] == img_id), None)
            if raw_img_info is None:
                continue
            raw_img_info['img_id'] = img_id

            ann_ids = [ann['id'] for ann in self.urpc.get('annotations', []) if ann['image_id'] == img_id]
            raw_ann_info = [ann for ann in self.urpc.get('annotations', []) if ann['image_id'] == img_id]
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    # def __getitem__(self, idx):
    #     # 从原始数据信息中获取 img_path 和 instances
    #     data_info = self.get_data_info(idx)
    #     img_path = data_info['img_path']
    #     img_id = data_info['img_id']  # 获取 img_id
    #     instances = data_info['instances']
        
    #     # 提取 GT 信息
    #     gt_bboxes = []
    #     gt_labels = []
    #     for instance in instances:
    #         gt_bboxes.append(instance['bbox'])
    #         gt_labels.append(instance['bbox_label'])
        
    #     # 转换为张量
    #     gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
    #     gt_labels = torch.tensor(gt_labels, dtype=torch.int64)
        
    #     # 执行数据处理管道，确保保留 img_path 和 img_id
    #     processed_data = self.prepare_data(idx)
    #     processed_data['img_path'] = img_path
    #     processed_data['img_id'] = img_id  # 添加 img_id
        
    #     # 执行数据处理管道
    #     result = self.pipeline(processed_data)
        
    #     # 添加 GT 信息到处理后的结果中
    #     result['img_path'] = img_path
    #     result['img_id'] = img_id  # 添加 img_id
    #     result['gt_instances'] = {
    #         'bboxes': gt_bboxes,
    #         'labels': gt_labels
    #     }
        
    #     return result