#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch.fx
import torchvision.transforms as transforms
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config, TQTSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode, TQTThresholdInitMeth
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerTensorPowOf2MinMSEObserver, PerTensorPowOf2MinMaxObserver
from quark.torch.quantization.observer.tqt_observer import TQTObserver
from quark.torch.quantization.observer.lsq_observer import LSQObserver
# from torch._export import capture_pre_autograd_graph
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/group/modelzoo/test_dataset/Imagenet/', help='Data set directory.')
parser.add_argument('--model_name', default='resnet18', choices=['mobilenetv2', 'resnet18'], help='Model to be used.')
parser.add_argument('--pretrained', default=None, help='Pre trained model weights')
parser.add_argument('--qat', action="store_true", help='Perform QAT to further improve accuracy.')
parser.add_argument('--tqt', action="store_true", help='Perform TQT to further improve accuracy.')
parser.add_argument('--lsq', action="store_true", help='Perform LSQ to further improve accuracy.')
parser.add_argument('--non_overflow', action="store_true", help='Perform non overflow quantizer to perform PTQ/QAT.')
parser.add_argument('--mse_powof2', action="store_true", help='Perform mse_powof2 quantizer to perform PTQ/QAT.')
parser.add_argument('--early_stop', action="store_true", help='During training whether to early stop.')
parser.add_argument('--early_stop_step', default=2, type=int, help='Condition to early stop.')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=3, type=int, help='Training epochs.')
parser.add_argument('--quantizer_lr', default=1e-5, type=float, help='Initial lr rate: quantizer param (For TQT).')
parser.add_argument('--quantizer_lr_decay', default=0.5, type=float, help='Learning rate decay ratio of quantizer.')
parser.add_argument('--weight_lr', default=1e-5, type=float, help='Initial learning rate of network weights.')
parser.add_argument('--weight_lr_decay', default=0.94, type=int, help='Learning rate decay ratio of network weights.')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument('--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument('--val_batch_size', default=128, type=int, help='Batch size for validation.')
parser.add_argument('--display_freq', default=100, type=int, help='Display training metrics every n steps.')
parser.add_argument('--val_freq', default=500, type=int, help='Validate model every n steps.')
parser.add_argument('--save_dir', default='./quant_result', help='Directory to save trained models.')
parser.add_argument('--weight_lr_decay_steps', type=int, default=2000, help='adjust learning rate: newwork params')
parser.add_argument('--quantizer_lr_decay_steps', type=int, default=1000, help='adjust learning rate: quantizer params')
parser.add_argument('--gpus', type=str, default='0', help='gpu ids to be used for training, seperated by commas')
parser.add_argument('--quant_ckpt', default=None, help='Dir to save model state_dict.')
parser.add_argument("--model_export",
                    help="Model export format",
                    default=None,
                    action="append",
                    choices=[None, "torch_compile", "onnx", "torch_save"])
parser.add_argument('--export_dir', default=None, help='Dir to save export onnx model.')
args, _ = parser.parse_known_args()


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


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def prepare_calib_dataset(data_path, device='cpu', calib_length=10000):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valdir = os.path.join(data_path, 'validation')
    dataset_test = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    def calibration_collate_fn(batch):
        '''Dataset for calibration, without label
        '''
        inputs, _ = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        return inputs

    dataset_test = Subset(dataset_test, list(range(min(calib_length, len(dataset_test)))))
    collate_fn = calibration_collate_fn
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test,
                                  batch_size=args.val_batch_size,
                                  sampler=test_sampler,
                                  collate_fn=collate_fn,
                                  num_workers=0)
    return data_loader_test


def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'validation')

    dataset_test = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.val_batch_size,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.train_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    return data_loader, data_loader_test


def train_one_step(model, inputs, criterion, optimizer, device):
    images, target = inputs
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    # compute gradient and do paramupdate step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc1, acc5


def validate(val_loader, model, criterion, device, full_test=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model = model.to(device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.display_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg


def save_directory():
    return os.path.join(args.save_dir, args.model_name)


def save_checkpoint(state, is_best, directory):
    # mkdir_if_not_exist
    if directory and (not os.path.isdir(directory)):
        os.makedirs(directory)
    if not os.path.isdir(directory):
        raise RuntimeError("Failed to create dir %r" % directory)

    filepath = os.path.join(directory, 'model.pth')
    torch.save(state, filepath)
    if is_best:
        best_acc1 = state['best_acc1'].item()
        best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
        shutil.copyfile(filepath, best_filepath)
        print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))
    return best_filepath if is_best else filepath


def adjust_learning_rate(optimizer, epoch, step):
    """Sets the learning rate to the initial LR decayed by decay ratios"""
    for param_group in optimizer.param_groups:
        group_name = param_group['name']
        if group_name == 'weight' and step != 0 and step % args.weight_lr_decay_steps == 0:
            old_lr = param_group['lr']
            lr = args.weight_lr * (args.weight_lr_decay**(step / args.weight_lr_decay_steps))
            param_group['lr'] = lr
            print('Adjust weight lr, epoch {}, step {}: group_name={}, old lr={}, new lr={}'.format(
                epoch, step, group_name, old_lr, lr))
        if group_name == 'quantizer' and step != 0 and step % args.quantizer_lr_decay_steps == 0:
            old_lr = param_group['lr']
            lr = args.quantizer_lr * (args.quantizer_lr_decay**(step / args.quantizer_lr_decay_steps))
            param_group['lr'] = lr
            print('Adjust quantizer lr epoch {}, step {}: group_name={}, old lr={}, new lr={}'.format(
                epoch, step, group_name, old_lr, lr))


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


def quantizer_parameters(model):
    params = []
    for module in model.modules():
        if isinstance(module, TQTObserver):
            params += [module._log_threshold]
    return params


def non_quantizer_parameters(model):
    params = []
    quantizer_parameters_ids = set([id(x) for x in quantizer_parameters(model)])
    for param in model.parameters():
        if id(param) not in quantizer_parameters_ids:
            params.append(param)
    return params


def train(model, train_loader, val_loader, criterion, device_ids):
    best_acc1 = 0
    best_filepath = None
    if args.early_stop is True:
        not_improve_term = 0

    if device_ids is not None and len(device_ids) > 0:
        device = f"cuda:{device_ids[0]}"
        model = model.to(device)
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
    if device_ids is None:
        device = 'cpu'

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    param_groups = [{
        'params':
        quantizer_parameters(model) if not isinstance(model, nn.DataParallel) else quantizer_parameters(model.module),
        'lr':
        args.quantizer_lr,
        'name':
        'quantizer'
    }, {
        'params':
        non_quantizer_parameters(model)
        if not isinstance(model, nn.DataParallel) else non_quantizer_parameters(model.module),
        'lr':
        args.weight_lr,
        'name':
        'weight'
    }]

    optimizer = torch.optim.Adam(param_groups, args.weight_lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs):
        progress = ProgressMeter(len(train_loader) * args.epochs, [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch[{}], Step: ".format(epoch))

        for i, (images, target) in enumerate(train_loader):
            end = time.time()
            # measure data loading time
            data_time.update(time.time() - end)
            step = len(train_loader) * epoch + i

            adjust_learning_rate(optimizer, epoch, step)
            loss, acc1, acc5 = train_one_step(model, (images, target), criterion, optimizer, device)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if step % args.display_freq == 0:
                progress.display(step)

            if step % args.val_freq == 0:
                # evaluate on validation set
                print('epoch: {}, step: {}'.format(epoch, i))
                acc1 = validate(val_loader, model, criterion, device)
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                filepath = save_checkpoint(
                    {
                        'epoch':
                        epoch + 1,
                        'state_dict':
                        model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                        'best_acc1':
                        best_acc1
                    }, is_best, save_directory())
                if is_best:
                    best_filepath = filepath
                # early stop session
                if args.early_stop is True:
                    not_improve_term = (not_improve_term + 1) if not is_best else 0
                    print("is bset: ", is_best, "not_improve_term:", not_improve_term)
                    if not_improve_term >= args.early_stop_step:
                        print("As not improved, finish training")
                        model.load_state_dict(torch.load(best_filepath)['state_dict'])
                        return best_filepath
    model.load_state_dict(torch.load(best_filepath)['state_dict'])
    return best_filepath


def load_model(model_name):
    assert model_name in ["mobilenetv2", "resnet18"], "model must be one of [mobilenetv2, resnet18]"
    from torchvision.models import resnet18, mobilenet_v2
    model = resnet18(pretrained=False) if model_name == "resnet18" else mobilenet_v2(pretrained=False)
    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict)
    return model


def get_graph_module(float_model, example_inputs):
    '''
    Using PyTorch official API to get the GraphModule
    '''
    logger.info("Start to capture program...")
    try:
        model = torch.export.export_for_training(float_model.eval(), example_inputs).module()
        logger.info("Get graph module successfully.")
        return model
    except Exception as e:
        logger.exception(f"Pytorch internal error: {str(e)}")


def main():
    print('Used arguments:', args)
    device_ids = None if args.gpus == "" else [int(i) for i in args.gpus.split(",")]
    device = f"cuda:{device_ids[0]}" if device_ids is not None and len(device_ids) > 0 else "cpu"
    if device_ids is None:
        device = 'cpu'

    # Init dummy input, dataset, float model
    example_inputs = (torch.rand(args.train_batch_size, 3, 224, 224).to(device), )
    calib_loader = prepare_calib_dataset(args.data_dir, device, calib_length=args.val_batch_size * 2)
    train_loader, val_loader = prepare_data_loaders(args.data_dir)
    float_model = load_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.tqt or args.lsq:
        assert args.qat, "Must set qat is True!"

    # Init quantization config and instance quantizer
    INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)
    if args.tqt:
        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_WEIGHT = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=TQTObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
            qat_spec=TQTSpec(threshold_init_meth=TQTThresholdInitMeth._3SD))

        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_INPUT = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=TQTObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
            qat_spec=TQTSpec(threshold_init_meth=TQTThresholdInitMeth._KL_J))
        quant_config = QuantizationConfig(weight=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_WEIGHT,
                                          input_tensors=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_INPUT,
                                          output_tensors=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_INPUT,
                                          bias=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_WEIGHT)
        calib_loader = []  # if using tqt, we will directly train, skip PTQ
    elif args.lsq:
        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_WEIGHT = QuantizationSpec(dtype=Dtype.int8,
                                                                       qscheme=QSchemeType.per_tensor,
                                                                       observer_cls=LSQObserver,
                                                                       symmetric=True,
                                                                       scale_type=ScaleType.float,
                                                                       round_method=RoundType.half_even,
                                                                       is_dynamic=False)

        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT = QuantizationSpec(dtype=Dtype.int8,
                                                                      qscheme=QSchemeType.per_tensor,
                                                                      observer_cls=LSQObserver,
                                                                      symmetric=True,
                                                                      scale_type=ScaleType.float,
                                                                      round_method=RoundType.half_even,
                                                                      is_dynamic=False)
        quant_config = QuantizationConfig(weight=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_WEIGHT,
                                          input_tensors=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT,
                                          output_tensors=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT,
                                          bias=DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_WEIGHT)
    elif args.non_overflow:
        INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                qscheme=QSchemeType.per_tensor,
                                                observer_cls=PerTensorPowOf2MinMaxObserver,
                                                symmetric=True,
                                                scale_type=ScaleType.float,
                                                round_method=RoundType.half_even,
                                                is_dynamic=False)
        quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                          output_tensors=INT8_PER_TENSOR_SPEC,
                                          weight=INT8_PER_TENSOR_SPEC,
                                          bias=INT8_PER_TENSOR_SPEC)
    elif args.mse_powof2:
        INT8_PER_WEIGHT_TENSOR_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorPowOf2MinMSEObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False)

        INT8_PER_ACTIVTION_TENSOR_SPEC = QuantizationSpec(
            dtype=Dtype.uint8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorPowOf2MinMSEObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False)

        quant_config = QuantizationConfig(input_tensors=INT8_PER_ACTIVTION_TENSOR_SPEC,
                                          output_tensors=INT8_PER_ACTIVTION_TENSOR_SPEC,
                                          weight=INT8_PER_WEIGHT_TENSOR_SPEC,
                                          bias=INT8_PER_WEIGHT_TENSOR_SPEC)
    else:
        quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                          output_tensors=INT8_PER_TENSOR_SPEC,
                                          weight=INT8_PER_TENSOR_SPEC,
                                          bias=INT8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
    quantizer = ModelQuantizer(quant_config)
    # prepare the torch.fx.GraphModule
    graph_model = get_graph_module(float_model, example_inputs)
    # optimize GraphModule and insert quantizer
    quantized_model = quantizer.quantize_model(graph_model, calib_loader)
    # Test the validation accuracy after PTQ
    print("Evaluate the validation accuracy after PTQ:")
    if not args.tqt:
        acc1 = validate(val_loader, quantized_model, criterion, device)
    # User can train the model (QAT) to further improve accuracy.
    if args.qat is True:
        train(quantized_model, train_loader, val_loader, criterion, device_ids)

    # Currently, model export may not support LSQ.
    if args.model_export is not None:
        # checkpoint = './qat_models/{MODEL_NAME}}/model_best_{***}.pth'
        # quantized_model.load_state_dict(torch.load(checkpoint)['state_dict'])
        freezeded_model = quantizer.freeze(quantized_model.eval())
        if "onnx" in args.model_export:
            from quark.torch import ModelExporter
            from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
            config = ExporterConfig(json_export_config=JsonExporterConfig())
            exporter = ModelExporter(config=config, export_dir=args.export_dir)
            # NOTE for NPU compile, it is better using batch-size = 1 for better compliance
            example_inputs = (torch.rand(1, 3, 224, 224).to(device), )
            exporter.export_onnx_model(freezeded_model, example_inputs[0])
        if "torch_save" in args.model_export:
            # save session
            from quark.torch.export.api import save_params
            example_inputs = (next(iter(val_loader))[0].to(device), )
            save_params(freezeded_model,
                        model_type=args.model_name,
                        args=example_inputs,
                        export_dir=args.export_dir,
                        quant_mode=quant_config.quant_mode)
            # example_inputs = (next(iter(val_loader))[0].to(device), )
            # model_file_path = os.path.join(args.export_dir, args.model_name + "_quantized.pth")
            # exported_model = torch.export.export(freezeded_model, example_inputs)
            # torch.export.save(exported_model, model_file_path)
        if "torch_compile" in args.model_export:
            print("\nCalling PyTorch 2 torch.compile...")
            # Note: The model after torch.compile may not be able to export to other format
            freezeded_model = torch.compile(freezeded_model)


if __name__ == '__main__':
    main()
