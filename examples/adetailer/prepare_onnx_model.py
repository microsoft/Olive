import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from pathlib import Path

def download(model_name: str):
	dir = Path("./models", model_name.split("_")[0])
	dir.mkdir(parents=True, exist_ok=True)

	hf_hub_download("Bingsu/adetailer", f"{model_name}.pt", local_dir= f"./{dir}/")

	# 加载YOLOv8模型
	yolo_model = YOLO(f'{dir}/{model_name}.pt')

	torch_model = yolo_model.model
	torch.save(torch_model, f"{dir}/{model_name}_pytorch.pt")

	# 将模型导出为ONNX格式
	success = yolo_model.export(format='onnx')

	# 检查导出是否成功
	if success:
		print("模型成功转换为ONNX格式。")
	else:
		print("模型转换失败。")

download("face_yolov9c")
download("hand_yolov9c")
download("person_yolov8m-seg")
download("deepfashion2_yolov8s-seg")