import os

# common file definition
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'vgg16')
TEST_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp-pytorch', 'input')
PRETRAINED_MODEL_VIDEO_DATA = os.path.join(TEST_INPUT_DIR, 'random_fake_data_video_r2plus1d_18.npz')
TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torch_home')


CLASSIFICATION_MODEL_NAMES = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                              'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
                              'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn',
                              'vgg19', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3', 'densenet121',
                              'densenet169', 'densenet201', 'densenet161', 'googlenet', 'mobilenet_v2',
                              'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5',
                              'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
CLASSIFICATION_MODEL_DONOT_PRETRAIN = ['mnasnet0_75', 'mnasnet1_3', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']

# customized model
CUSTOMIZED_MODEL_MULTIPLE_INPUTS_DATA = os.path.join(TEST_INPUT_DIR, 'customized_model_multiple_inputs.npz')

CUSTOMIZED_MODEL_MULTIPLE_OUTPUTS_DATA = os.path.join(TEST_INPUT_DIR, 'customized_model_multiple_outputs.npz')

CUSTOMIZED_MODEL_MULTIPLE_INPUTS_OUTPUTS_DATA = os.path.join(TEST_INPUT_DIR,
                                                             'customized_model_multiple_inputs_outputs.npz')
