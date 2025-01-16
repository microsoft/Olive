# download src image + imagenet_classes.txt + _label.txt + _data.npz
from datasets import load_dataset
from pathlib import Path
import numpy as np
import os
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def download_dataset(data_dir: Path, split: str, size: int):
    image_dir = data_dir / str(split)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset in streaming mode to avoid downloading the full dataset
    dataset = load_dataset(
        'imagenet-1k',
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    label_file = os.path.join(image_dir, '_labels.txt')

    labels = []
    images = []
    with open(label_file, 'w') as f:
        for i, sample in enumerate(dataset):
            if i >= size:
                break

            image = sample['image']
            label = sample['label']

            image_path = os.path.join(image_dir, f'{i}.jpg')
            image.save(image_path)

            image = image.convert('RGB')
            image = preprocess(image)

            images.append(image)
            labels.append(label)

            # save image source file
            # f.write(f'{i}.jpg,{label}\n')

    print(f"Images saved to {image_dir} and labels saved to {label_file}.")
    np.savez(os.path.join(image_dir, '_data.npz'), images=np.array(images), labels=np.array(labels))

# Download ImageNet labels
import urllib
def download_classfile(data_dir: Path):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classFile = os.path.join(data_dir, "imagenet_classes.txt")
    urllib.request.urlretrieve(url, classFile)

import torch
import os
def download_resnet_model(model_path: Path, model_name: str):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    torch.save(model, model_path)