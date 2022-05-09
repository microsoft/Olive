import os
import urllib.request
import uuid

from PIL import Image
import numpy as np
import pytest
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from ci_tests.tests.conversion.multiple_io_model import MultipleInputsModel, MultipleOutputsModel, MultipleInputAndOutputsModel
from ci_tests.tests.conversion.constants import *


def get_img_tensor(img_path, img_size=None):
    img = Image.open(img_path)
    min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    if img_size:
        min_img_size = img_size
    transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                             transforms.CenterCrop(min_img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    img = Variable(img)

    return img

def ensure_path_exist(path):
    dir_path = os.path.dirname(path)
    ensure_dir_exists(dir_path)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def prepare_test_data():
    # prepare test input data
    vgg16_test_cat_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'vgg16', 'cat.jpg')
    vgg16_test_husky_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'vgg16', 'husky.jpg')

    # download if not exist
    if not os.path.exists(vgg16_test_cat_img_path):
        ensure_path_exist(vgg16_test_cat_img_path)
        urllib.request.urlretrieve(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg',
            vgg16_test_cat_img_path)
    if not os.path.exists(vgg16_test_husky_img_path):
        ensure_path_exist(vgg16_test_husky_img_path)
        urllib.request.urlretrieve(
            "http://cdn.shopify.com/s/files/1/0994/0236/articles/siberian-husky_1200x1200.jpg?v=1502391918",
            vgg16_test_husky_img_path)

    # save test data to disk
    cat_img_244 = get_img_tensor(vgg16_test_cat_img_path, 244)

    if not os.path.exists(PRETRAINED_MODEL_VIDEO_DATA):
        ensure_path_exist(PRETRAINED_MODEL_VIDEO_DATA)
        value_list = [torch.randn(2, 3, 4, 112, 112).data.numpy()]
        name_list = ["input_0"]
        data = dict(zip(name_list, value_list))
        np.savez(PRETRAINED_MODEL_VIDEO_DATA, **data)

    if not os.path.exists(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA):
        ensure_path_exist(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA)
        x = np.random.randn(10, 3, 244, 244)
        value_list = [x, x, x]
        name_list = ["multiple_in_0", "multiple_in_1", "multiple_in_2"]
        data = dict(zip(name_list, value_list))
        np.savez(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA, **data)

    if not os.path.exists(CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA):
        ensure_path_exist(CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA)
        x = np.random.randn(10, 3, 244, 244)
        value_list = [x]
        name_list = ["multiple_in_0"]
        data = dict(zip(name_list, value_list))
        np.savez(CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA, **data)

    if not os.path.exists(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA):
        ensure_path_exist(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA)
        x = np.random.randn(10, 3, 244, 244)
        value_list = [x, x]
        name_list = ["multiple_in_0", "multiple_in_1"]
        data = dict(zip(name_list, value_list))
        np.savez(CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA, **data)

def setup_torch_cache_folder():
    # setup torchvision download cache folder
    torch_home = os.path.join(TORCH_HOME, f"_{uuid.uuid4()}")
    os.environ['TORCH_HOME'] = torch_home

    if not os.path.exists(torch_home):
        os.makedirs(torch_home)


def prepare_model(model_path, model_type, inputs_num=None):
    setup_torch_cache_folder()
    model = None
    if model_type in CLASSIFICATION_MODEL_NAMES:
        pretrained = False if model_type in CLASSIFICATION_MODEL_DONOT_PRETRAIN else True
        model = models.__dict__[model_type](pretrained=pretrained, progress=False)
    elif model_type == "segmentation_deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
    elif model_type == "segmentation_fcn_resnet101":
        model = models.segmentation.fcn_resnet101(pretrained=True, progress=False)
    elif model_type == "video_r2plus1d_18":
        model = models.video.r2plus1d_18(pretrained=True, progress=False)
    elif model_type == "video_mc3_18":
        model = models.video.mc3_18(pretrained=True, progress=False)
    elif model_type == "video_r3d_18":
        model = models.video.r3d_18(pretrained=True, progress=False)
    elif model_type == "customized_model_multiple_inputs":
        model = MultipleInputsModel()
    elif model_type == "customized_model_multiple_outputs":
        model = MultipleOutputsModel()
    elif model_type == "customized_model_multiple_inputs_outputs":
        model = MultipleInputAndOutputsModel()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.eval()
    torch.save(model, model_path)


def get_classification_params():
    params = []
    for name in range(len(CLASSIFICATION_MODEL_NAMES)):
        params.append(pytest.param(CLASSIFICATION_MODEL_NAMES[name]))
    return params


def save_test_data_to_disk(path):
    title_lengths = np.array([8]).astype(np.int32)
    title_encoder = np.random.random_sample([8, 1, 300]).astype(np.float32)
    ratings = np.array([0]).astype(np.int32)
    query_lengths = np.array([2]).astype(np.int32)
    passage_lengths = np.array([60]).astype(np.int32)
    features = np.random.random_sample([1, 19]).astype(np.float32)
    encoder = np.random.random_sample([60, 1, 300]).astype(np.float32)
    decoder = np.random.random_sample([2, 1, 300]).astype(np.float32)
    placeholder = np.array(0.0, np.float32)
    test_data = [title_lengths, title_encoder, ratings, query_lengths, passage_lengths, features, encoder, decoder,
                 placeholder]

    name_list = ["title_lengths:0", "title_encoder:0", "ratings:0", "query_lengths:0",
                 "passage_lengths:0", "features:0", "encoder:0", "decoder:0", "Placeholder:0"]
    data = dict(zip(name_list, test_data))
    # save to file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)
