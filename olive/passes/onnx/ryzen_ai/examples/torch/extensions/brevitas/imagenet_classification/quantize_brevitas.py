"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Implements the Brevitas quantization config and quantizer shim based on Quark interface.
This file should live outside of Quark codebase.
"""

from datasets import load_dataset, Dataset
from tqdm import tqdm
from functools import partial
import argparse
from torch.utils.data import DataLoader

from brevitas.quant_tensor import QuantTensor
import quark.torch.extensions.brevitas.config as brevitas_config
import quark.torch.extensions.brevitas.api as brevitas_api
import torch
import torchvision

from datasets.features.image import Image
from pathlib import Path
from imagenet_classes import IMAGENET2012_CLASSES
from typing import Optional
parser = argparse.ArgumentParser(prog='ResNet-50 example')
parser.add_argument("--quant_scheme",
                    help="Supported quant_scheme in the script. If there is no suitable quantization strategy among the options, users can customize the quantization configuration according to their own needs.",
                    default="w_int8_per_tensor_sym",
                    choices=["w_int8_per_tensor_sym", "w_int8_a_int8_per_tensor_sym"])

parser.add_argument('--calibration_samples', type=int, default=128, help="Number of samples from the dataset train set to use in case activations are statically quantized.")
parser.add_argument('--evaluation_samples', type=int, default=None, help="Limit the evaluation to a certain number of samples, useful in case the full imagenet validation set is not available on disk.")
parser.add_argument("--no_eval_reference", action="store_true", default=False, help="Disable reference model evaluation.")

parser.add_argument('--dataset', default="imagenet-1k", help='Image dataset to use for (optional) calibration and evaluation.')
parser.add_argument('--data_dir', default=None, help="Path to a directory storing the dataset")
parser.add_argument('--model', default='resnet50', choices=['resnet18', 'resnet50'], help='Model to be used from torchvision.')
parser.add_argument('--validation_batch_size', default=16, type=int, help='Batch size for validation.')
args, _ = parser.parse_known_args()


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def collate_fn(batch_data, processor):
    # Some imagenet images are in greyscale.
    pixel_values = torch.stack([processor(data["image"].convert('RGB')) for data in batch_data])
    label = torch.tensor([data["label"] for data in batch_data])

    return (pixel_values, label)

def validate(val_loader, model):
    """
    Run validation on the desired dataset
    """
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader)):
            target = target.to(device)
            target = target.to(dtype)
            images = images.to(device)
            images = images.to(dtype)

            output = model(images)
            if isinstance(output, QuantTensor):
                output = output.value
            # measure accuracy
            acc1, = accuracy(output, target)
            top1.update(acc1[0], output.size(0))

        print(f'Total: Avg acc@1 {top1.avg:2.3f}')
    return top1.avg.cpu().numpy()

def prepare_validation_data(dataset_name: str, data_dir: Optional[str]):
    if data_dir and dataset_name != "imagenet-1k":
        raise NotImplementedError(f"Loading a local validation set is only supported for imagenet-1k (got {dataset_name}).")

    dataset = load_dataset(Path(data_dir, "validation").as_posix(), split="validation").cast_column("image", Image(decode=False))

    image_ids = []
    imagenet_classes = list(IMAGENET2012_CLASSES.keys())
    for data in tqdm(dataset):
        filename = Path(data["image"]["path"]).stem
        class_id = filename.split("_")[3]

        # We simply refer to the class index, and do not rely on the trained model label2id, see https://huggingface.co/datasets/ILSVRC/imagenet-1k/discussions/1
        image_id = imagenet_classes.index(class_id)
        image_ids.append(image_id)

    dataset = dataset.cast_column("image", Image(decode=True))
    dataset = dataset.add_column("label", image_ids)

    return dataset


def main():
    args = parser.parse_args()

    print("Loading model...")
    if args.model == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading calibration dataset...")
    # Avoid loading calibration dataset in case it is not needed.
    if args.quant_scheme == "w_int8_a_int8_per_tensor_sym":
        # Use streaming to avoid the need to download the full dataset.
        iterable_dataset = load_dataset(args.dataset, split='train', streaming=True, token=True)
        iterable_dataset = iterable_dataset.shuffle(seed=42)

        data = []
        iterator = iter(iterable_dataset)
        for _ in range(args.calibration_samples):
            data.append(next(iterator))

        calibration_dataset = Dataset.from_list(data)

        calibration_dataloader = DataLoader(calibration_dataset, batch_size=1, collate_fn=partial(collate_fn, processor=processor))
    else:
        calibration_dataloader = None

    print("Loading validation dataset...")
    if args.evaluation_samples is None:
        if args.data_dir is None:
            # HF datasets downloads all splits despite specifying a split, see https://github.com/huggingface/datasets/issues/6793, it is preferable
            # to download offline the validation split.
            validation_set = load_dataset(args.data_dir, split="validation")
        else:
            validation_set = prepare_validation_data(args.dataset, args.data_dir)
    else:
        # We may limit the number of evaluation sample for quick testing, using streaming, allowing to avoid to download the full validation dataset.
        iterable_validation_set = load_dataset(args.dataset, split='validation', streaming=True, token=True)
        iterable_validation_set = iterable_validation_set.shuffle(seed=42)

        data = []
        iterator = iter(iterable_validation_set)
        for _ in range(args.evaluation_samples):
            data.append(next(iterator))

        validation_set = Dataset.from_list(data)

    validation_dataloader = DataLoader(validation_set, batch_size=args.validation_batch_size, collate_fn=partial(collate_fn, processor=processor))

    if not args.no_eval_reference:
        print("Evaluating the original float32 model...")
        validate(validation_dataloader, model)

    weight_spec = brevitas_config.QuantizationSpec()
    input_spec = None
    output_spec = None

    if args.quant_scheme == "w_int8_a_int8_per_tensor_sym":
        input_spec = brevitas_config.QuantizationSpec()

    global_config = brevitas_config.QuantizationConfig(weight=weight_spec, input_tensors=input_spec, output_tensors=output_spec)
    config = brevitas_config.Config(global_quant_config=global_config, pre_quant_opt_config=[])

    quantizer = brevitas_api.ModelQuantizer(config)

    print("Quantizing the model...")
    quantized_model = quantizer.quantize_model(model, calibration_dataloader)

    print("Evaluating the quantized model...")
    validate(validation_dataloader, quantized_model)


if __name__ == "__main__":
    main()
