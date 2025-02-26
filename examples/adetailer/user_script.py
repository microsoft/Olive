# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import torch
from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from olive.data.registry import Registry
from olive.model import OliveModelHandler

logger = getLogger(__name__)

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from torch import from_numpy
from torch.utils.data import Dataset
import torchvision.ops as ops
from olive.data.registry import Registry
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logger = getLogger(__name__)

_curlabels_np = None

class FaceDataset(Dataset):	
	def __init__(self, data):
		global _curlabels_np
		self.images_np = data['images']
		_curlabels_np = data['labels']
	def __len__(self):
		return min(len(self.images_np), len(_curlabels_np))

	def __getitem__(self, idx):
		input_img = self.images_np[idx]
		input_img = np.transpose(input_img, (2, 0, 1))
		input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0
		return {"images": input_img}, torch.tensor([idx], dtype=torch.int32)
	
def face_get_boxes(output):
	# 后处理结果
	confidence_threshold = 0.1
	count = 0
	boxes = []
	scores = []

	# 遍历每个预测框
	for i in range(output.shape[1]):
		# 提取置信度和类别概率
		confidence = output[4, i]
		if confidence > confidence_threshold:
			count += 1
			# 提取边界框坐标
			x_center = output[0, i]
			y_center = output[1, i]
			width = output[2, i]
			height = output[3, i]

			# 计算边界框的左上角和右下角坐标
			x1 = int(x_center - width / 2)
			y1 = int(y_center - height / 2)
			x2 = int(x_center + width / 2)
			y2 = int(y_center + height / 2)
			boxes.append([x1, y1, x2, y2])
			scores.append(confidence.item())

	if len(boxes) == 0:
		return boxes, scores

	boxes = torch.tensor(boxes, dtype=torch.float32)
	scores = torch.tensor(scores, dtype=torch.float32)	

	# 应用非极大值抑制
	nms_threshold = 0.4
	keep_indices = ops.nms(boxes, scores, nms_threshold)
	keep_indices = keep_indices.tolist()

	keep_boxes = []
	keep_scores = []
	for i in keep_indices:
		x1, y1, x2, y2 = boxes[i].int().tolist()  # 将坐标转换为普通的Python整数列表		
		keep_boxes.append([x1, y1, x2, y2])
		keep_scores.append(scores[i])
	
	return keep_boxes, keep_scores

@Registry.register_pre_process()
def face_pre_process(validation_dataset, **kwargs):
	cache_key = kwargs.get("cache_key")
	size = kwargs.get("size", 256)
	cache_file = None
	if cache_key:
		cache_file = Path(f"./cache/data/{cache_key}_{size}.npz")
		if cache_file.exists():
			with np.load(Path(cache_file), allow_pickle=True) as data:
				return FaceDataset(data)

	# 初始化空列表来存储不同的数据字段
	images = []
	labels = []

	# 定义目标图像大小
	target_size = (640, 640)

	for i, sample in enumerate(validation_dataset):
		if i == size:
			break
		# 将PIL.Image转换为numpy数组
		saved_img = sample["image"]
		original_width, original_height = saved_img.size
		# 调整图像大小
		saved_img = saved_img.resize(target_size)
		img_array = np.array(saved_img)
		images.append(img_array)

		bbox_list = sample["faces"]["bbox"]
		scaled_bbox_list = []
		# 计算宽度和高度的缩放比例
		width_scale = target_size[0] / original_width
		height_scale = target_size[1] / original_height
		for bbox in bbox_list:
			x, y, w, h = bbox
			# 对边界框坐标进行缩放
			scaled_x = x * width_scale
			scaled_y = y * height_scale
			scaled_w = w * width_scale
			scaled_h = h * height_scale
			scaled_bbox_list.append([scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h])
		labels.append(scaled_bbox_list)

	images_np = np.array(images)
	labels_np = np.array(labels, dtype=object)
	result_data = {"images": images_np, "labels": labels_np}

	if cache_file:
		cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
		np.savez(cache_file, **result_data)

	return FaceDataset(result_data)


def eval_accuracy(model: OliveModelHandler, device, execution_providers, batch_size=1, **kwargs):
	# dataset = face_dataset()
	# dataloader = DataLoader(dataset, batch_size)
	# preds = []
	# target = []
	# sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)

	# for inputs, labels in dataloader:
	# 	res = model.run_session(sess, inputs)
	# 	if len(output_names) == 1:
	# 		result = torch.Tensor(res[0])
	# 	else:
	# 		result = torch.Tensor(res)
	# 	outputs = bert_post_process(result)
	# 	preds.extend(outputs.tolist())
	# 	target.extend(labels.data.tolist())
	return


def face_metric(model_output, targets):
	global _curlabels_np
	prediction_data = []
	target_data = []

	for i, target in enumerate(targets):
		keep_boxes, keep_scores = face_get_boxes(model_output[0][i])
		target_boxes = _curlabels_np[target]
		prediction_data.append({
			'boxes': torch.tensor(keep_boxes, dtype= torch.float32),
			'scores': torch.tensor(keep_scores, dtype=torch.float32),
			'labels': torch.zeros(len(keep_boxes), dtype=torch.int64),
		})
		target_data.append({
			'boxes': torch.tensor(target_boxes, dtype= torch.float32),
			'labels': torch.zeros(len(target_boxes), dtype=torch.int64),
		})

	# 初始化 mAP 计算对象
	iou_thresholds = torch.arange(0.5, 1, 0.05).tolist()
	metric = MeanAveragePrecision(iou_thresholds = iou_thresholds)

	# 更新指标
	metric.update(prediction_data, target_data)

	# 计算 mAP
	result = metric.compute()
	return {
		"map 50-95": result["map"],
		"map 50": result["map_50"]
	}
